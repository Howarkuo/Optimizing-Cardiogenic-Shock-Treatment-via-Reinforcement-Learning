import duckdb
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tianshou.data import Batch, ReplayBuffer
from tianshou.utils.net.common import Net
from tianshou.policy import DQNPolicy
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & GPU SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mimic_shock.db")
PHENO_FILE = os.path.join(BASE_DIR, "phenotype_results", "Compensated_ids.csv")

# TIMESTAMP FOR UNIQUE LOGS/PLOTS
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")
LOG_FILE = os.path.join(BASE_DIR, f"training_pipeline_{TIMESTAMP}.log")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

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
    
    # Keep ALL Compensated Shock patients (including zero-dose)
    df = df[df['stay_id'].isin(comp_ids)] 
    df = df.sort_values(['stay_id', 'chart_hour']).reset_index(drop=True)
    
    logging.info(f"Filtering complete. Found {len(df)} patient records.")

    # --- ACTION MAPPING ---
    logging.info("Mapping 5 Vasopressor Actions...")
    active_vaso = df[df['vaso_rate'] > 0]['vaso_rate']
    
    if not active_vaso.empty:
        vaso_bins = [0] + list(active_vaso.quantile([0.25, 0.5, 0.75, 1.0]))
    else:
        vaso_bins = [0, 0.1, 0.2, 0.3, 0.4] 
    
    def get_action(val):
        if val <= 0: return 0
        for i in range(len(vaso_bins)-1):
            if val <= vaso_bins[i+1]: return i + 1
        return 4
    df['action'] = df['vaso_rate'].apply(get_action)

    # --- REWARD CALCULATION (Physiological Distress Score) ---
    logging.info("Calculating State-Based Rewards (Physiological Distress)...")
    map_distress = np.where(df['map'] < 65, 2, np.where(df['map'] > 90, 1, 0))
    hr_distress = np.where(df['hr'] > 110, 1, 0)
    lac_distress = np.where(df['lactate'] > 2.0, 2, 0)
    
    df['physio_distress'] = map_distress + hr_distress + lac_distress
    # df['reward'] = df['physio_distress'] - df.groupby('stay_id')['physio_distress'].shift(-1)
    # bug in code: fill the NaN values before applying the terminal mortality rewards for shift (-1), in pandas if phys - NaN = Nan
    # must .fillna(0)
    # Calculate distress improvement, filling the final step's NaN with 0 first
    df['reward'] = (df['physio_distress'] - df.groupby('stay_id')['physio_distress'].shift(-1)).fillna(0)
    
    # NOW apply the terminal reward
    is_last_step = df.groupby('stay_id').cumcount(ascending=False) == 0
    df.loc[is_last_step, 'reward'] += df.loc[is_last_step, 'mortality'].map({0: 15, 1: -15})
    # is_last_step = df.groupby('stay_id').cumcount(ascending=False) == 0
    # df.loc[is_last_step, 'reward'] += df.loc[is_last_step, 'mortality'].map({0: 15, 1: -15})
    df['reward'] = df['reward'].fillna(0)

    # --- NEXT STATES & DONE FLAGS ---
    for col in FEATURES:
        df[f'next_{col}'] = df.groupby('stay_id')[col].shift(-1).fillna(0)
    df['done'] = is_last_step.astype(int)

    # --- NORMALIZATION ---
    logging.info("Normalizing Features...")
    for col in ['lactate', 'creatinine', 'vaso_rate', 'urine_output']:
        df[col] = np.log1p(np.clip(df[col], a_min=0, a_max=None))
        df[f'next_{col}'] = np.log1p(np.clip(df[f'next_{col}'], a_min=0, a_max=None))
    
    train_ids, val_ids = train_test_split(df['stay_id'].unique(), test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(df[df['stay_id'].isin(train_ids)][FEATURES])
    
    # Added .values to suppress Sklearn warnings
    df[FEATURES] = scaler.transform(df[FEATURES].values)
    next_features = [f'next_{c}' for c in FEATURES]
    df[next_features] = scaler.transform(df[next_features].values)
    
    con.close()
    return df, train_ids, val_ids

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

    logging.info("Loading data into ReplayBuffer...")
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
    logging.info("Starting Training Epochs...")
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
        
        if epoch % 100 == 0:
            logging.info(f"Epoch {epoch:4d} | Loss: {current_loss:.4f} | Q-Avg: {current_q:.4f} | Eps: {eps:.2f}")

    return policy, q_vals, losses

# ==========================================
# 4. EVALUATION & PLOTTING (OPE)
# ==========================================
def evaluate_and_plot(policy, df, val_ids, q_vals):
    logging.info("Running Off-Policy Evaluation & Generating AI Predictions...")
    
    # --------------------------------------------------
    # STEP 1: RUN THE MODEL FIRST (Creates val_df & result)
    # --------------------------------------------------
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
    
    # --------------------------------------------------
    # STEP 2: GENERATE ALL PLOTS
    # --------------------------------------------------
    logging.info("Generating Plots...")

    # --- PLOT: D3QN Convergence ---
    plt.figure(figsize=(10, 5))
    plt.plot(q_vals)
    plt.title('D3QN Convergence (Compensated Subgroup)')
    plt.ylabel('Mean Q-Value (V)')
    plt.xlabel('Training Steps (Epochs)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(BASE_DIR, f"q_convergence_{TIMESTAMP}.png"))
    plt.close()

    # --- PLOT: Treatment Action Distributions ---
    logging.info("Generating Treatment Distribution Plot...")
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
    plt.xlabel('Vasopressor Action Bins (0 = No Drug, 4 = Max Drug)')
    plt.ylabel('Frequency (%)')
    plt.xticks(x, ['Action 0', 'Action 1', 'Action 2', 'Action 3', 'Action 4'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"action_distribution_{TIMESTAMP}.png"))
    plt.close()

    # --- PLOT: Estimated Policy Value (OPE) ---
    logging.info("Calculating Estimated Policy Values via Q-Estimates...")
    q_vals_matrix = result.logits.cpu().numpy()
    val_df['q_ai'] = q_vals_matrix.max(axis=1)
    val_df['q_clinician'] = q_vals_matrix[np.arange(len(val_df)), val_df['action']]
    val_df['q_zero'] = q_vals_matrix[:, 0]
    val_df['q_random'] = q_vals_matrix.mean(axis=1)

    patient_returns = val_df.groupby('stay_id')[['q_clinician', 'q_ai', 'q_zero', 'q_random']].mean()

    plt.figure(figsize=(8, 6))
    box = plt.boxplot(
        [patient_returns['q_clinician'], patient_returns['q_ai'], patient_returns['q_zero'], patient_returns['q_random']],
        patch_artist=True,
        labels=['Clinicians', 'AI Policy', 'Zero Drug', 'Random Policy'],
        showfliers=False 
    )
    colors = ['lightcoral', 'dodgerblue', 'lightgray', 'khaki']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title('Estimated Policy Value Comparison (Validation Set)')
    plt.ylabel('Expected Return (Average Q-Value per Patient)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"policy_value_ope_{TIMESTAMP}.png"))
    plt.close()

    # --- PLOT: Mortality vs. AI Policy Deviation ---
    patient_summary = val_df.groupby('stay_id').agg({'action_gap': 'mean', 'mortality': 'first'})
    patient_summary['gap_bin'] = pd.cut(patient_summary['action_gap'], 
                                        bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
                                        labels=['Much Less', 'Less', 'Match', 'More', 'Much More'])
    mort_rate = patient_summary.groupby('gap_bin', observed=False)['mortality'].mean() * 100
    
    plt.figure(figsize=(8, 6))
    mort_rate.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Mortality vs. AI Policy Deviation (Compensated Shock)')
    plt.ylabel('Mortality Rate (%)')
    plt.xlabel('Clinician Dose vs AI Recommendation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"ope_mortality_curve_{TIMESTAMP}.png"))
    plt.close()

    # --- PLOT: Mortality Risk vs. Return ---
    logging.info("Generating Mortality Risk vs. Return Plot...")
    patient_returns['mortality'] = val_df.groupby('stay_id')['mortality'].first()
    
    try:
        patient_returns['return_decile'] = pd.qcut(patient_returns['q_clinician'], q=20, duplicates='drop')
        bin_stats = patient_returns.groupby('return_decile', observed=False).agg(
            mean_return=('q_clinician', 'mean'),
            mortality_rate=('mortality', 'mean')
        ).reset_index()

        plt.figure(figsize=(8, 6))
        plt.plot(bin_stats['mean_return'], bin_stats['mortality_rate'], marker='o', linestyle='-', color='indigo', linewidth=2)
        plt.axhline(y=patient_returns['mortality'].mean(), color='gray', linestyle=':', label='Average Baseline Mortality')
        plt.title('Mortality Risk vs. AI Calculated Return')
        plt.xlabel('Return of Actions (AI Q-Value of Actual Treatment)')
        plt.ylabel('Actual Mortality Risk (Probability)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, f"mortality_vs_return_{TIMESTAMP}.png"))
        plt.close()
    except Exception as e:
        logging.warning(f"Could not generate Mortality vs Return plot: {e}")

    # --- PLOT: Return Distributions by Survival ---
    logging.info("Generating Return Distributions by Survival Plot...")
    survivors_return = patient_returns[patient_returns['mortality'] == 0]['q_clinician']
    nonsurvivors_return = patient_returns[patient_returns['mortality'] == 1]['q_clinician']

    plt.figure(figsize=(8, 6))
    plt.hist(survivors_return, bins=30, alpha=0.6, color='dodgerblue', edgecolor='black', density=True, label='Survivors')
    plt.hist(nonsurvivors_return, bins=30, alpha=0.6, color='crimson', edgecolor='black', density=True, label='Nonsurvivors')
    plt.title('Distribution of Average Return per Patient')
    plt.xlabel('Average Return per Patient (Q-Value)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"return_distribution_{TIMESTAMP}.png"))
    plt.close()
    
    logging.info(f"All Plots Saved to {BASE_DIR} with tag {TIMESTAMP}")
# ==========================================
# 5. MAIN EXECUTION WITH ERROR CATCHING
# ==========================================
if __name__ == "__main__":
    try:
        logging.info("========================================")
        logging.info("Starting D3QN Pipeline Execution")
        logging.info("========================================")
        
        full_df, train_idx, val_idx = prepare_clinical_data()
        model_policy, q_history, loss_history = train_d3qn(full_df, train_idx)
        evaluate_and_plot(model_policy, full_df, val_idx, q_history)
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error("A fatal error occurred during execution:", exc_info=True)

# 2026-04-02 12:08:00,307 - INFO - ========================================
# 2026-04-02 12:08:00,307 - INFO - Starting D3QN Pipeline Execution
# 2026-04-02 12:08:00,307 - INFO - ========================================
# 2026-04-02 12:08:00,308 - INFO - Connecting to Database and Loading Data...
# 2026-04-02 12:08:00,807 - INFO - Filtering complete. Found 192246 patient records.
# 2026-04-02 12:08:00,807 - INFO - Mapping 5 Vasopressor Actions...
# 2026-04-02 12:08:00,903 - INFO - Calculating State-Based Rewards (Physiological Distress)...
# 2026-04-02 12:08:01,048 - INFO - Normalizing Features...
# 2026-04-02 12:08:01,256 - INFO - Initializing D3QN on cuda...
# 2026-04-02 12:08:01,431 - INFO - Loading data into ReplayBuffer...
# 2026-04-02 12:08:10,798 - INFO - Starting Training Epochs...
# 2026-04-02 12:08:12,104 - INFO - Epoch  100 | Loss: 0.9684 | Q-Avg: 0.1948 | Eps: 0.80
# 2026-04-02 12:08:13,074 - INFO - Epoch  200 | Loss: 0.9194 | Q-Avg: 0.2264 | Eps: 0.60
# 2026-04-02 12:08:14,043 - INFO - Epoch  300 | Loss: 0.9015 | Q-Avg: 0.1384 | Eps: 0.40
# 2026-04-02 12:08:14,961 - INFO - Epoch  400 | Loss: 0.8592 | Q-Avg: 0.0952 | Eps: 0.20
# 2026-04-02 12:08:15,915 - INFO - Epoch  500 | Loss: 0.8941 | Q-Avg: 0.2109 | Eps: 0.05
# 2026-04-02 12:08:16,858 - INFO - Epoch  600 | Loss: 0.5792 | Q-Avg: 0.1760 | Eps: 0.05
# 2026-04-02 12:08:17,779 - INFO - Epoch  700 | Loss: 0.6403 | Q-Avg: 0.2675 | Eps: 0.05
# 2026-04-02 12:08:18,714 - INFO - Epoch  800 | Loss: 0.6783 | Q-Avg: 0.2579 | Eps: 0.05
# 2026-04-02 12:08:19,652 - INFO - Epoch  900 | Loss: 0.5766 | Q-Avg: 0.2348 | Eps: 0.05
# 2026-04-02 12:08:20,570 - INFO - Epoch 1000 | Loss: 0.6494 | Q-Avg: 0.2192 | Eps: 0.05
# 2026-04-02 12:08:20,572 - INFO - Running Off-Policy Evaluation & Generating AI Predictions...
# 2026-04-02 12:08:20,627 - INFO - Generating Plots...
# 2026-04-02 12:08:20,864 - INFO - Generating Treatment Distribution Plot...
# 2026-04-02 12:08:21,059 - INFO - Calculating Estimated Policy Values via Q-Estimates...
# 2026-04-02 12:08:21,423 - INFO - Generating Mortality Risk vs. Return Plot...
# 2026-04-02 12:08:21,642 - INFO - Generating Return Distributions by Survival Plot...
# 2026-04-02 12:08:21,933 - INFO - All Plots Saved to /home/howard900126 with tag 0402_1208
# 2026-04-02 12:08:21,933 - INFO - Pipeline completed successfully!
