import duckdb
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import random
import pickle
import json
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tianshou.data import Batch, ReplayBuffer
from tianshou.utils.net.common import Net
from tianshou.policy import DQNPolicy
from datetime import datetime

# ==========================================
# 1. CONFIGURATION, SEEDING & GPU SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mimic_shock.db")
PHENO_FILE = os.path.join(BASE_DIR, "phenotype_results", "Compensated_ids.csv")

TIMESTAMP = datetime.now().strftime("%m%d_%H%M")
LOG_FILE = os.path.join(BASE_DIR, f"training_pipeline_{TIMESTAMP}.log")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

# Fix random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

FEATURES = [
    'sbp', 'map', 'hr', 'resp', 'temp', 'spo2', 'cvp', 'pcwp', 
    'lactate', 'creatinine', 'glucose', 'potassium', 'sodium', 
    'ph', 'po2', 'pco2', 'fio2', 'urine_output', 'vaso_rate', 'age'
]

LR = 1e-4
GAMMA = 0.99
EPS_MIN = 0.05
BATCH_SIZE = 128
EPISODES = 1000

# ==========================================
# 2. DATA PREPROCESSING & ACTION MAPPING
# ==========================================
def prepare_clinical_data():
    logging.info("Connecting to Database and Loading Data...")
    con = duckdb.connect(DB_PATH)
    
    comp_ids = pd.read_csv(PHENO_FILE)['stay_id'].tolist()
    df = con.execute("SELECT * FROM rl_state_space").fetchdf()
    df_labels = con.execute("SELECT stay_id, hospital_expire_flag as mortality FROM cohort").fetchdf()
    df = df.merge(df_labels, on='stay_id', how='left')
    
    df = df[df['stay_id'].isin(comp_ids)] 
    df = df.sort_values(['stay_id', 'chart_hour']).reset_index(drop=True)
    logging.info(f"Filtering complete. Found {len(df)} patient records.")

    # --- ACTION MAPPING & REPORTING ---
    active_vaso = df[df['vaso_rate'] > 0]['vaso_rate']
    if not active_vaso.empty:
        vaso_bins = [0] + list(active_vaso.quantile([0.25, 0.5, 0.75, 1.0]))
    else:
        vaso_bins = [0, 0.1, 0.2, 0.3, 0.4] 
    
    # Log the exact clinical dose ranges for transparency
    bin_ranges = [f"{vaso_bins[i]:.3f} - {vaso_bins[i+1]:.3f}" for i in range(len(vaso_bins)-1)]
    logging.info(f"Action Bins Dose Ranges: Action 0 = 0.0, Action 1-4 = {bin_ranges}")
    
    def get_action(val):
        if val <= 0: return 0
        for i in range(len(vaso_bins)-1):
            if val <= vaso_bins[i+1]: return i + 1
        return 4
    df['action'] = df['vaso_rate'].apply(get_action)

    # --- REWARD CALCULATION ---
    map_distress = np.where(df['map'] < 65, 2, np.where(df['map'] > 90, 1, 0))
    hr_distress = np.where(df['hr'] > 110, 1, 0)
    lac_distress = np.where(df['lactate'] > 2.0, 2, 0)
    
    df['physio_distress'] = map_distress + hr_distress + lac_distress
    df['reward'] = (df['physio_distress'] - df.groupby('stay_id')['physio_distress'].shift(-1)).fillna(0)
    
    is_last_step = df.groupby('stay_id').cumcount(ascending=False) == 0
    # Increased terminal reward to prevent short-sighted hacking
    df.loc[is_last_step, 'reward'] += df.loc[is_last_step, 'mortality'].map({0: 50, 1: -50})

    # --- NEXT STATES & DONE FLAGS ---
    for col in FEATURES:
        df[f'next_{col}'] = df.groupby('stay_id')[col].shift(-1).fillna(0)
    df['done'] = is_last_step.astype(int)

    # --- SAFE NORMALIZATION ---
    logging.info("Normalizing Features safely...")
    for col in ['lactate', 'creatinine', 'vaso_rate', 'urine_output']:
        df[col] = np.log1p(np.clip(df[col], a_min=0, a_max=None))
        df[f'next_{col}'] = np.log1p(np.clip(df[f'next_{col}'], a_min=0, a_max=None))
    
    # train_ids, val_ids = train_test_split(df['stay_id'].unique(), test_size=0.2, random_state=SEED)
    
    # scaler = StandardScaler()
    # # Apply correctly aligned Pandas transform to avoid Sklearn shape errors
    # df[FEATURES] = scaler.fit_transform(df[FEATURES])
    
    # next_features = [f'next_{c}' for c in FEATURES]
    # next_df_aligned = df[next_features].rename(columns={f'next_{c}': c for c in FEATURES})
    # df[next_features] = scaler.transform(next_df_aligned)
    # Split IDs first
    train_ids, val_ids = train_test_split(df['stay_id'].unique(), test_size=0.2, random_state=SEED)
    
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    scaler.fit(df[df['stay_id'].isin(train_ids)][FEATURES])
    
    # Transform BOTH using the training-only scaler
    df[FEATURES] = scaler.transform(df[FEATURES])
    
    # Do the same for next_features
    next_features = [f'next_{c}' for c in FEATURES]
    next_df_aligned = df[next_features].rename(columns={f'next_{c}': c for c in FEATURES})
    df[next_features] = scaler.transform(next_df_aligned)
    
    con.close()
    return df, train_ids, val_ids, scaler, vaso_bins

# ==========================================
# 3. D3QN ARCHITECTURE & TRAINING
# ==========================================
def train_d3qn(df, train_ids):
    logging.info(f"Initializing D3QN on {DEVICE}...")
    train_df = df[df['stay_id'].isin(train_ids)].copy()
    
    buffer = ReplayBuffer(size=len(train_df))
    next_features = [f'next_{c}' for c in FEATURES]
    
    obs = train_df[FEATURES].values.astype(np.float32)
    act = train_df['action'].values
    rew = train_df['reward'].values.astype(np.float32)
    obs_next = train_df[next_features].values.astype(np.float32)
    dones = train_df['done'].values.astype(bool)

    for i in trange(len(train_df), desc="Filling Buffer", unit="step"):
        buffer.add(Batch(
            obs=obs[i], act=act[i], rew=rew[i], 
            done=dones[i], terminated=dones[i], truncated=False, 
            obs_next=obs_next[i], info={}
        ))

    net = Net(state_shape=len(FEATURES), action_shape=5, hidden_sizes=[256, 256], device=DEVICE,
              dueling_param=({"hidden_sizes": [128]}, {"hidden_sizes": [128]})).to(DEVICE)
    
    policy = DQNPolicy(model=net, optim=torch.optim.Adam(net.parameters(), lr=LR),
                       discount_factor=GAMMA, estimation_step=3, target_update_freq=500, is_double=True)

    q_vals, losses = [] , []
    pbar = trange(1, EPISODES + 1, desc="Training D3QN", unit="ep")
    
    for epoch in pbar:
        stats = policy.update(sample_size=BATCH_SIZE, buffer=buffer)
        current_loss = stats.get('loss', 0)

        batch, _ = buffer.sample(BATCH_SIZE)
        with torch.no_grad():
            current_q = policy(batch).logits.max(dim=1)[0].mean().item()
            
        q_vals.append(current_q)
        losses.append(current_loss)
        
        eps = max(EPS_MIN, 1.0 - (epoch / 500))
        policy.set_eps(eps)
        pbar.set_postfix(loss=f"{current_loss:.4f}", q_avg=f"{current_q:.4f}", eps=f"{eps:.2f}")

    return policy, q_vals, losses

# ==========================================
# 4. EVALUATION & PLOTTING 
# ==========================================
def evaluate_and_plot(policy, df, val_ids, q_vals, losses):
    logging.info("Generating AI Predictions & Evaluating...")
    
    val_df = df[df['stay_id'].isin(val_ids)].copy()
    policy.eval()
    val_batch = Batch(obs=val_df[FEATURES].values.astype(np.float32), info={})
    
    with torch.no_grad():
        result = policy(val_batch)
        ai_actions = result.logits.argmax(axis=1)
        if isinstance(ai_actions, torch.Tensor):
            ai_actions = ai_actions.cpu().numpy()
            
    val_df['ai_action'] = ai_actions
    val_df['action_gap'] = val_df['action'] - val_df['ai_action']
    
    # --- PLOT 1 & 2: Convergence & Loss ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(losses, color='crimson')
    axes[0].set_title('Training Loss')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(q_vals, color='dodgerblue')
    axes[1].set_title('Mean Q-Value Convergence')
    axes[1].set_ylabel('Q-Value')
    axes[1].set_xlabel('Epochs')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"training_metrics_{TIMESTAMP}.png"))
    plt.close()
    # --- ADD THESE TWO LINES HERE ---
    plot_research_curves(val_df, n_bootstrap=100)
    plot_classic_red_trajectory(val_df, TIMESTAMP, BASE_DIR)


    # --- PLOT 3: Treatment Action Distributions ---
    clinician_dist = val_df['action'].value_counts(normalize=True).sort_index() * 100
    ai_dist = val_df['ai_action'].value_counts(normalize=True).sort_index() * 100
    for i in range(5):
        if i not in clinician_dist: clinician_dist[i] = 0
        if i not in ai_dist: ai_dist[i] = 0
            
    x = np.arange(5)
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, clinician_dist, width, label='Clinician', color='lightcoral', edgecolor='black')
    plt.bar(x + width/2, ai_dist, width, label='AI Policy', color='dodgerblue', edgecolor='black')
    plt.title('Vasopressor Action Distribution: Clinician vs AI')
    plt.ylabel('Frequency (%)')
    plt.xticks(x, ['Action 0', 'Action 1', 'Action 2', 'Action 3', 'Action 4'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(BASE_DIR, f"action_distribution_{TIMESTAMP}.png"))
    plt.close()

    # --- PLOT 4: Q-Estimated Policy Value (Renamed from OPE) ---
    logging.info("Calculating Model-Implied Q-Estimates...")
    q_vals_matrix = result.logits.cpu().numpy()
    val_df['q_ai'] = q_vals_matrix.max(axis=1)
    val_df['q_clinician'] = q_vals_matrix[np.arange(len(val_df)), val_df['action']]
    val_df['q_zero'] = q_vals_matrix[:, 0]
    val_df['q_random'] = q_vals_matrix.mean(axis=1)

    patient_returns = val_df.groupby('stay_id')[['q_clinician', 'q_ai', 'q_zero', 'q_random']].mean()

    plt.figure(figsize=(8, 6))
    box = plt.boxplot(
        [patient_returns['q_clinician'], patient_returns['q_ai'], patient_returns['q_zero'], patient_returns['q_random']],
        patch_artist=True, labels=['Clinicians', 'AI Policy', 'Zero Drug', 'Random Policy'], showfliers=False 
    )
    for patch, color in zip(box['boxes'], ['lightcoral', 'dodgerblue', 'lightgray', 'khaki']):
        patch.set_facecolor(color)
    plt.title('Q-Estimated Policy Value Comparison (Model Proxy)')
    plt.ylabel('Expected Return (Average Q-Value per Patient)')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(BASE_DIR, f"q_estimated_value_{TIMESTAMP}.png"))
    plt.close()

    # --- PLOT 5: The "U-Curve" - Mortality vs Dose Difference ---
    logging.info("Generating Mortality vs Dose Difference Plot...")

    patient_summary = val_df.groupby('stay_id').agg({
        'action_gap': 'mean',
        'mortality': 'first'
    })

    # Labels represent the Clinician's dose relative to the AI's recommendation
    patient_summary['gap_bin'] = pd.cut(
        patient_summary['action_gap'], 
        bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
        labels=['Clinician Much Less', 'Clinician Less', 'Match', 'Clinician More', 'Clinician Much More']
    )

    mort_stats = patient_summary.groupby('gap_bin', observed=False).agg(
        rate=('mortality', 'mean'),
        n=('mortality', 'count')
    )
    mort_stats['rate'] *= 100

    plt.figure(figsize=(10, 6))
    ax = mort_stats['rate'].plot(kind='bar', color='skyblue', edgecolor='black', zorder=3)
    
    plt.title('Patient Mortality vs. Vasopressor Dosing Deviation', fontsize=14)
    plt.xlabel('Clinician Action relative to AI Recommendation', fontsize=12)
    plt.ylabel('In-Hospital Mortality Rate (%)', fontsize=12)
    plt.xticks(rotation=25)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Annotate n-values to check for data support
    for i, row in enumerate(mort_stats.itertuples()):
        if not np.isnan(row.rate):
            ax.text(i, row.rate + 1, f'n={int(row.n)}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"mortality_u_curve_{TIMESTAMP}.png"))
    plt.close()

def plot_research_curves(val_df, n_bootstrap=100):
    logging.info(f"Generating Bootstrap Patient-Clustered Curves (N={n_bootstrap})...")
    
    # 1. Define Discrete Bins centered on integers [-4, -3, ..., 0, ..., 4]
    bins = np.arange(-4.5, 5.5, 1.0)
    bin_centers = np.arange(-4, 5, 1.0)
    
    def get_curve_stats(data):
        # Calculate mortality rate per bin
        data['bin'] = pd.cut(data['action_gap'], bins=bins)
        stats = data.groupby('bin', observed=False)['mortality'].mean()
        return stats.values

    # ---------------------------------------------------------
    # STEP 1: PREPARE DATA
    # ---------------------------------------------------------
    unique_ids = val_df['stay_id'].unique()
    
    # Patient-level dataframe
    pat_df = val_df.groupby('stay_id').agg({'action_gap': 'mean', 'mortality': 'first'})
    
    # ---------------------------------------------------------
    # STEP 2: BOOTSTRAP LOOP (The "Manual Bootstrap")
    # ---------------------------------------------------------
    boot_pat_means = []
    boot_step_means = []
    
    for _ in range(n_bootstrap):
        # Sample patient IDs with replacement
        sample_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
        
        # Patient-Level Sample
        # Since IDs repeat, we use a trick to create a new dataframe
        boot_pat_sample = pat_df.loc[sample_ids]
        boot_pat_means.append(get_curve_stats(boot_pat_sample))
        
        # Step-Level Sample (Clustered)
        # We must take ALL rows for the sampled patients
        boot_step_sample = pd.concat([val_df[val_df['stay_id'] == sid] for sid in sample_ids])
        boot_step_means.append(get_curve_stats(boot_step_sample))

    # ---------------------------------------------------------
    # STEP 3: COMPUTE PERCENTILES (95% CI)
    # ---------------------------------------------------------
    def finalize_plot_data(boot_list, original_data):
        boot_array = np.array(boot_list)
        mean_val = np.nanmean(boot_array, axis=0)
        lower_ci = np.nanpercentile(boot_array, 2.5, axis=0)
        upper_ci = np.nanpercentile(boot_array, 97.5, axis=0)
        return mean_val, lower_ci, upper_ci

    p_mean, p_lower, p_upper = finalize_plot_data(boot_pat_means, pat_df)
    s_mean, s_lower, s_upper = finalize_plot_data(boot_step_means, val_df)

    # ---------------------------------------------------------
    # STEP 4: PLOTTING
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    for ax, m, l, u, title in zip(axes, [p_mean, s_mean], [p_lower, s_lower], 
                                   [p_upper, s_upper], ['Patient-Level (Average Stay)', 'Step-Level (Local Action)']):
        # Plot the mean line
        ax.plot(bin_centers, m, color='navy', marker='o', linewidth=2, zorder=3)
        # Plot the bootstrap CI ribbon
        ax.fill_between(bin_centers, l, u, color='crimson', alpha=0.2, label='95% Bootstrap CI')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Action Gap (Clinician - AI)')
        ax.set_ylabel('In-Hospital Mortality Rate')
        ax.set_xticks(bin_centers)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"research_mortality_curves_{TIMESTAMP}.png"))
    plt.close()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_classic_red_trajectory(val_df, timestamp, base_dir):
    logging.info("Generating Classic Red Trajectory Plot...")
    
    # 1. Define Integer Bins (Centering on -2, -1, 0, 1, 2)
    # We ignore the extreme +/- 3 or 4 if data is too sparse
    bin_centers = np.array([-2, -1, 0, 1, 2])
    bins = np.arange(-2.5, 3.5, 1.0)
    
    unique_ids = val_df['stay_id'].unique()
    boot_means = []
    
    # 2. Clustered Bootstrap (Resample Patients, not Rows)
    for _ in range(100):
        sample_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
        # Efficiently collect all rows for sampled patients
        boot_sample = pd.concat([val_df[val_df['stay_id'] == sid] for sid in sample_ids])
        
        # Calculate mortality per bin
        boot_sample['bin'] = pd.cut(boot_sample['action_gap'], bins=bins)
        # Use observed=False to ensure we get a value for every bin
        m_rate = boot_sample.groupby('bin', observed=False)['mortality'].mean().values
        boot_means.append(m_rate)
    
    # 3. Process Bootstrap Results
    boot_array = np.array(boot_means)
    mean_trajectory = np.nanmean(boot_array, axis=0)
    lower_bound = np.nanpercentile(boot_array, 2.5, axis=0)
    upper_bound = np.nanpercentile(boot_array, 97.5, axis=0)
    
    # 4. Plotting the "Nature Style" Figure
    plt.figure(figsize=(8, 6))
    
    # The Shaded Confidence Ribbon
    plt.fill_between(bin_centers, lower_bound, upper_bound, color='red', alpha=0.2, label='95% CI')
    
    # The Main Red Trajectory Line
    plt.plot(bin_centers, mean_trajectory, color='darkred', linewidth=2.5, marker='o', markersize=4, label='Mean Mortality')
    
    # Formatting to match high-impact journals
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='AI Match')
    plt.title('Clinical Outcome vs. Policy Deviation', fontsize=14, fontweight='bold')
    plt.xlabel('Vasopressor Dose Difference (Clinician - AI)', fontsize=12)
    plt.ylabel('In-Hospital Mortality Rate', fontsize=12)
    plt.xticks(bin_centers)
    plt.ylim(0, 1.0) # Scale to 0-100% or use actual range
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    plt.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"classic_red_trajectory_{timestamp}.png"), dpi=300)
    plt.close()
    logging.info("Red Trajectory Plot Saved.")
# ==========================================
# 5. MAIN EXECUTION & ARTIFACT SAVING
# ==========================================
if __name__ == "__main__":
    try:
        logging.info("Starting Research-Grade D3QN Pipeline Execution")
        
        full_df, train_idx, val_idx, scaler, vaso_bins = prepare_clinical_data()
        model_policy, q_history, loss_history = train_d3qn(full_df, train_idx)
        evaluate_and_plot(model_policy, full_df, val_idx, q_history, loss_history)
        
        # --- SAVE ARTIFACTS FOR REPRODUCIBILITY ---
        logging.info("Saving Model and Artifacts...")
        torch.save(model_policy.state_dict(), os.path.join(BASE_DIR, f"d3qn_weights_{TIMESTAMP}.pth"))
        
        with open(os.path.join(BASE_DIR, f"scaler_{TIMESTAMP}.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
            
        config_data = {
            "vaso_bins": vaso_bins,
            "features": FEATURES,
            "train_patients": len(train_idx),
            "val_patients": len(val_idx),
            "seed": SEED
        }
        with open(os.path.join(BASE_DIR, f"config_{TIMESTAMP}.json"), 'w') as f:
            json.dump(config_data, f, indent=4)
            
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error("A fatal error occurred during execution:", exc_info=True)




# (rl_env) howard900126@dhlab-SVR-TS700-E9-RS8:~$ tail -f training_0424-2.log
# nohup: ignoring input
# 2026-04-24 00:57:41,928 - INFO - Starting Research-Grade D3QN Pipeline Execution
# 2026-04-24 00:57:41,928 - INFO - Connecting to Database and Loading Data...
# 2026-04-24 00:57:42,414 - INFO - Filtering complete. Found 192246 patient records.
# 2026-04-24 00:57:42,425 - INFO - Action Bins Dose Ranges: Action 0 = 0.0, Action 1-4 = ['0.000 - 0.050', '0.050 - 0.100', '0.100 - 0.150', '0.150 - 10.695']
# 2026-04-24 00:57:42,648 - INFO - Normalizing Features safely...
# 2026-04-24 00:57:42,924 - INFO - Initializing D3QN on cuda...
# Filling Buffer: 100%|██████████| 149659/149659 [00:06<00:00, 23420.81step/s]
# Training D3QN: 100%|██████████| 1000/1000 [00:12<00:00, 81.97ep/s, eps=0.05, loss=93.5364, q_avg=1.2712]
# 2026-04-24 00:58:04,663 - INFO - Generating AI Predictions & Evaluating...
# 2026-04-24 00:58:05,182 - INFO - Generating Bootstrap Patient-Clustered Curves (N=100)...
# resample-newplot-008a_d3qn_gpu_new_0dose_newreward-newfig0423.py:362: RuntimeWarning: Mean of empty slice
#   mean_val = np.nanmean(boot_array, axis=0)
# /home/howard900126/rl_env/lib/python3.8/site-packages/numpy/lib/nanfunctions.py:1577: RuntimeWarning: All-NaN slice encountered
#   result = np.apply_along_axis(_nanquantile_1d, axis, a, q,
# 2026-04-24 00:58:29,270 - INFO - Generating Classic Red Trajectory Plot...
# 2026-04-24 00:58:52,709 - INFO - Red Trajectory Plot Saved.
# 2026-04-24 00:58:52,857 - INFO - Calculating Model-Implied Q-Estimates...
# 2026-04-24 00:58:52,993 - INFO - Generating Mortality vs Dose Difference Plot...
# 2026-04-24 00:58:53,216 - INFO - Saving Model and Artifacts...
# 2026-04-24 00:58:53,221 - INFO - Pipeline completed successfully!
# ^C
