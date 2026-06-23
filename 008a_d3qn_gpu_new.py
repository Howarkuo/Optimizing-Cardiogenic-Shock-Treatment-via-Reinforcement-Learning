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

# ==========================================
# 1. CONFIGURATION & GPU SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mimic_shock.db")
PHENO_FILE = os.path.join(BASE_DIR, "phenotype_results", "Compensated_ids.csv")
LOG_FILE = os.path.join(BASE_DIR, "training_pipeline.log")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup Logging (Writes to console AND a file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

FEATURES = [
    'sbp', 'map', 'hr', 'resp', 'temp', 'spo2', 'cvp', 'pcwp', 
    'lactate', 'creatinine', 'glucose', 'potassium', 'sodium', 
    'ph', 'po2', 'pco2', 'fio2', 'urine_output', 'vaso_rate', 'age'
]

# Hyperparameters (Thesis Specs)
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
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    if not os.path.exists(PHENO_FILE):
        raise FileNotFoundError(f"Phenotype file not found at {PHENO_FILE}")

    con = duckdb.connect(DB_PATH)
    
    comp_ids = pd.read_csv(PHENO_FILE)['stay_id'].tolist()
    df = con.execute("SELECT * FROM rl_state_space").fetchdf()
    df_labels = con.execute("SELECT stay_id, hospital_expire_flag as mortality FROM cohort").fetchdf()
    df = df.merge(df_labels, on='stay_id', how='left')
    
    df = df[df['stay_id'].isin(comp_ids) & (df.groupby('stay_id')['vaso_rate'].transform('max') > 0)]
    df = df.sort_values(['stay_id', 'chart_hour']).reset_index(drop=True)
    
    logging.info(f"Filtering complete. Found {len(df)} patient records.")

    # --- ACTION MAPPING ---
    logging.info("Mapping 5 Vasopressor Actions...")
    active_vaso = df[df['vaso_rate'] > 0]['vaso_rate']
    vaso_bins = [0] + list(active_vaso.quantile([0.25, 0.5, 0.75, 1.0]))
    
    def get_action(val):
        if val <= 0: return 0
        for i in range(len(vaso_bins)-1):
            if val <= vaso_bins[i+1]: return i + 1
        return 4
    df['action'] = df['vaso_rate'].apply(get_action)

    # --- REWARD CALCULATION ---
    logging.info("Calculating Rewards...")
    df['sofa_cv'] = np.where(df['vaso_rate'] > 0.1, 4, np.where(df['vaso_rate'] > 0, 3, 0))
    df['reward'] = df['sofa_cv'] - df.groupby('stay_id')['sofa_cv'].shift(-1)
    
    is_last_step = df.groupby('stay_id').cumcount(ascending=False) == 0
    df.loc[is_last_step, 'reward'] += df.loc[is_last_step, 'mortality'].map({0: 15, 1: -15})
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
    
    df[FEATURES] = scaler.transform(df[FEATURES])
    next_features = [f'next_{c}' for c in FEATURES]
    
    # Add .values to bypass the strict column name check
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
    
    # Extract native arrays for fast, crash-free loop execution
    obs = train_df[FEATURES].values
    act = train_df['action'].values
    rew = train_df['reward'].values
    obs_next = train_df[next_features].values
    dones = train_df['done'].values.astype(bool)

    logging.info("Loading data into ReplayBuffer...")
    for i in trange(len(train_df), desc="Filling Buffer", unit="step"):
        buffer.add(Batch(
            obs=obs[i],
            act=act[i],
            rew=rew[i],
            done=dones[i],
            terminated=dones[i],
            truncated=False,
            obs_next=obs_next[i],
            info={}
        ))

    net = Net(state_shape=len(FEATURES), action_shape=5, hidden_sizes=[256, 256], device=DEVICE,
              dueling_param=({"hidden_sizes": [128]}, {"hidden_sizes": [128]})).to(DEVICE)
    
    policy = DQNPolicy(model=net, optim=torch.optim.Adam(net.parameters(), lr=LR),
                       discount_factor=GAMMA, estimation_step=3, target_update_freq=500, is_double=True)

    q_vals, losses = [], []

    logging.info("Starting Training Epochs...")
    
    pbar = trange(1, EPISODES + 1, desc="Training D3QN", unit="ep")
    
    for epoch in pbar:
        # 1. Train the network
        stats = policy.update(sample_size=BATCH_SIZE, buffer=buffer)
        current_loss = stats.get('loss', 0)

        # 2. Manually calculate the Average Q-Value for a batch of data
        batch, _ = buffer.sample(BATCH_SIZE)
        with torch.no_grad():
            # policy(batch).logits outputs Q-values for all 5 actions. 
            # We take the max Q-value for each patient, then average them.
            current_q = policy(batch).logits.max(dim=1)[0].mean().item()
            
        q_vals.append(current_q)
        losses.append(current_loss)
        
        eps = max(EPS_MIN, 1.0 - (epoch / 500))
        policy.set_eps(eps)
        
        # Update the progress bar dynamically with live metrics
        pbar.set_postfix(loss=f"{current_loss:.4f}", q_avg=f"{current_q:.4f}", eps=f"{eps:.2f}")
        
        # Keep permanent log checkpoints every 100 epochs
        if epoch % 100 == 0:
            logging.info(f"Epoch {epoch:4d} | Loss: {current_loss:.4f} | Q-Avg: {current_q:.4f} | Eps: {eps:.2f}")

    return policy, q_vals, losses

# ==========================================
# 4. EVALUATION & PLOTTING (OPE)
# ==========================================
def evaluate_and_plot(policy, df, val_ids, q_vals):
    logging.info("Running Off-Policy Evaluation...")
    val_df = df[df['stay_id'].isin(val_ids)].copy()
    
    policy.eval()
    val_batch = Batch(obs=val_df[FEATURES].values, info={})
    
    with torch.no_grad():
        result = policy(val_batch)
        ai_actions = result.logits.argmax(axis=1)
        if isinstance(ai_actions, torch.Tensor):
            ai_actions = ai_actions.cpu().numpy()
            
    val_df['ai_action'] = ai_actions
    val_df['action_gap'] = val_df['action'] - val_df['ai_action']
    
    logging.info("Generating Plots...")
    plt.figure(figsize=(10, 5))
    plt.plot(q_vals)
    plt.title('D3QN Convergence (Compensated Subgroup)')
    plt.ylabel('Mean Q-Value (V)')
    plt.xlabel('Training Steps (Epochs)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(BASE_DIR, "q_convergence_2.png"))
    plt.close()

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
    plt.savefig(os.path.join(BASE_DIR, "ope_mortality_curve_2.png"))
    plt.close()
    
    logging.info(f"All Plots Saved to {BASE_DIR}")

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


# 2026-03-20 01:01:56,227 - INFO - ========================================
# 2026-03-20 01:01:56,227 - INFO -  Starting D3QN Pipeline Execution
# 2026-03-20 01:01:56,227 - INFO - ========================================
# 2026-03-20 01:01:56,227 - INFO - 📦 Connecting to Database and Loading Data...
# 2026-03-20 01:01:56,715 - INFO -  Filtering complete. Found 169888 patient records.
# 2026-03-20 01:01:56,715 - INFO -  Mapping 5 Vasopressor Actions...
# 2026-03-20 01:01:56,801 - INFO -  Calculating Rewards...
# 2026-03-20 01:01:56,920 - INFO -  Normalizing Features...
# 2026-03-20 01:01:57,100 - ERROR -  A fatal error occurred during execution:
# Traceback (most recent call last):
#   File "train_pipeline.py", line 225, in <module>
#     full_df, train_idx, val_idx = prepare_clinical_data()
#   File "train_pipeline.py", line 110, in prepare_clinical_data
#     df[next_features] = scaler.transform(df[next_features])
#   File "/home/howard900126/rl_env/lib/python3.8/site-packages/sklearn/utils/_set_output.py", line 157, in wrapped
#     data_to_wrap = f(self, X, *args, **kwargs)
#   File "/home/howard900126/rl_env/lib/python3.8/site-packages/sklearn/preprocessing/_data.py", line 1006, in transform
#     X = self._validate_data(
#   File "/home/howard900126/rl_env/lib/python3.8/site-packages/sklearn/base.py", line 580, in _validate_data
#     self._check_feature_names(X, reset=reset)
#   File "/home/howard900126/rl_env/lib/python3.8/site-packages/sklearn/base.py", line 507, in _check_feature_names
#     raise ValueError(message)
# ValueError: The feature names should match those that were passed during fit.
# Feature names unseen at fit time:
# - next_age
# - next_creatinine
# - next_cvp
# - next_fio2
# - next_glucose
# - ...
# Feature names seen at fit time, yet now missing:
# - age
# - creatinine
# - cvp
# - fio2
# - glucose
# - ...

# 2026-03-20 01:04:46,794 - INFO - ========================================
# 2026-03-20 01:04:46,794 - INFO -  Starting D3QN Pipeline Execution
# 2026-03-20 01:04:46,794 - INFO - ========================================
# 2026-03-20 01:04:46,794 - INFO - 📦 Connecting to Database and Loading Data...
# 2026-03-20 01:04:47,277 - INFO -  Filtering complete. Found 169888 patient records.
# 2026-03-20 01:04:47,278 - INFO -  Mapping 5 Vasopressor Actions...
# 2026-03-20 01:04:47,364 - INFO -  Calculating Rewards...
# 2026-03-20 01:04:47,483 - INFO -  Normalizing Features...
# 2026-03-20 01:04:47,656 - INFO -  Initializing D3QN on cuda...
# 2026-03-20 01:04:47,800 - ERROR -  A fatal error occurred during execution:
# Traceback (most recent call last):
#   File "train_pipeline.py", line 229, in <module>
#     model_policy, q_history, loss_history = train_d3qn(full_df, train_idx)
#   File "train_pipeline.py", line 128, in train_d3qn
#     buffer.add(Batch(
#   File "/home/howard900126/rl_env/lib/python3.8/site-packages/tianshou/data/buffer/base.py", line 257, in add
#     map(lambda x: np.array([x]), self._add_index(rew, done))
#   File "/home/howard900126/rl_env/lib/python3.8/site-packages/tianshou/data/buffer/base.py", line 209, in _add_index
#     if done:
# ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
# 2026-03-20 01:10:10,808 - INFO - ========================================
# 2026-03-20 01:10:10,808 - INFO -  Starting D3QN Pipeline Execution
# 2026-03-20 01:10:10,809 - INFO - ========================================
# 2026-03-20 01:10:10,809 - INFO -  Connecting to Database and Loading Data...
# 2026-03-20 01:10:11,295 - INFO -  Filtering complete. Found 169888 patient records.
# 2026-03-20 01:10:11,295 - INFO -  Mapping 5 Vasopressor Actions...
# 2026-03-20 01:10:11,381 - INFO -  Calculating Rewards...
# 2026-03-20 01:10:11,499 - INFO - Normalizing Features...
# 2026-03-20 01:10:11,702 - INFO -  Initializing D3QN on cuda...
# 2026-03-20 01:10:11,851 - INFO -  Loading data into ReplayBuffer...
# 2026-03-20 01:10:17,969 - INFO -  Starting Training Epochs...
# 2026-03-20 01:10:19,459 - INFO - Epoch  100 | Loss: 0.3959 | Q-Avg: 0.0000 | Eps: 0.80
# 2026-03-20 01:10:20,583 - INFO - Epoch  200 | Loss: 0.2396 | Q-Avg: 0.0000 | Eps: 0.60
# 2026-03-20 01:10:21,502 - INFO - Epoch  300 | Loss: 0.4954 | Q-Avg: 0.0000 | Eps: 0.40
# 2026-03-20 01:10:22,402 - INFO - Epoch  400 | Loss: 0.4071 | Q-Avg: 0.0000 | Eps: 0.20
# 2026-03-20 01:10:23,310 - INFO - Epoch  500 | Loss: 0.2670 | Q-Avg: 0.0000 | Eps: 0.05
# 2026-03-20 01:10:24,317 - INFO - Epoch  600 | Loss: 0.3632 | Q-Avg: 0.0000 | Eps: 0.05
# 2026-03-20 01:10:25,489 - INFO - Epoch  700 | Loss: 0.4463 | Q-Avg: 0.0000 | Eps: 0.05
# 2026-03-20 01:10:26,663 - INFO - Epoch  800 | Loss: 0.6021 | Q-Avg: 0.0000 | Eps: 0.05
# 2026-03-20 01:10:27,832 - INFO - Epoch  900 | Loss: 0.3359 | Q-Avg: 0.0000 | Eps: 0.05
# 2026-03-20 01:10:29,021 - INFO - Epoch 1000 | Loss: 0.1929 | Q-Avg: 0.0000 | Eps: 0.05
# 2026-03-20 01:10:29,023 - INFO -  Running Off-Policy Evaluation...
# 2026-03-20 01:10:29,087 - INFO -  Generating Plots...
# 2026-03-20 01:10:29,456 - INFO -  All Plots Saved to /home/howard900126
# 2026-03-20 01:10:29,457 - INFO -  Pipeline completed successfully!
# 2026-03-20 01:22:27,210 - INFO - ========================================
# 2026-03-20 01:22:27,210 - INFO - Starting D3QN Pipeline Execution
# 2026-03-20 01:22:27,210 - INFO - ========================================
# 2026-03-20 01:22:27,210 - INFO - Connecting to Database and Loading Data...
# 2026-03-20 01:22:27,701 - INFO - Filtering complete. Found 169888 patient records.
# 2026-03-20 01:22:27,701 - INFO - Mapping 5 Vasopressor Actions...
# 2026-03-20 01:22:27,787 - INFO - Calculating Rewards...
# 2026-03-20 01:22:27,903 - INFO - Normalizing Features...
# 2026-03-20 01:22:28,107 - INFO - Initializing D3QN on cuda...
# 2026-03-20 01:22:28,255 - INFO - Loading data into ReplayBuffer...
# 2026-03-20 01:22:36,510 - INFO - Starting Training Epochs...
# 2026-03-20 01:22:38,025 - INFO - Epoch  100 | Loss: 0.7569 | Q-Avg: 0.3785 | Eps: 0.80
# 2026-03-20 01:22:39,192 - INFO - Epoch  200 | Loss: 0.6198 | Q-Avg: 0.4199 | Eps: 0.60
# 2026-03-20 01:22:39,730 - INFO - Epoch  300 | Loss: 0.6135 | Q-Avg: 0.3902 | Eps: 0.40
# 2026-03-20 01:22:40,215 - INFO - Epoch  400 | Loss: 0.5857 | Q-Avg: 0.4781 | Eps: 0.20
# 2026-03-20 01:22:40,701 - INFO - Epoch  500 | Loss: 0.5852 | Q-Avg: 0.4251 | Eps: 0.05
# 2026-03-20 01:22:41,185 - INFO - Epoch  600 | Loss: 0.6808 | Q-Avg: 0.8490 | Eps: 0.05
# 2026-03-20 01:22:41,668 - INFO - Epoch  700 | Loss: 0.3374 | Q-Avg: 0.7822 | Eps: 0.05
# 2026-03-20 01:22:42,152 - INFO - Epoch  800 | Loss: 0.6900 | Q-Avg: 0.9815 | Eps: 0.05
# 2026-03-20 01:22:42,635 - INFO - Epoch  900 | Loss: 0.3619 | Q-Avg: 0.8584 | Eps: 0.05
# 2026-03-20 01:22:43,118 - INFO - Epoch 1000 | Loss: 0.2538 | Q-Avg: 0.9544 | Eps: 0.05
# 2026-03-20 01:22:43,119 - INFO - Running Off-Policy Evaluation...
# 2026-03-20 01:22:43,163 - INFO - Generating Plots...
# 2026-03-20 01:22:43,550 - INFO - All Plots Saved to /home/howard900126
# 2026-03-20 01:22:43,550 - INFO - Pipeline completed successfully!