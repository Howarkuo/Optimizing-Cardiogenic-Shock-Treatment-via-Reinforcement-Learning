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
    df['reward'] = df['physio_distress'] - df.groupby('stay_id')['physio_distress'].shift(-1)
    
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
    logging.info("Running Off-Policy Evaluation...")
    
    # --- THIS IS THE CRITICAL CODE YOU WERE MISSING ---
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

    logging.info("Generating Plots...")
    plt.figure(figsize=(10, 5))
    plt.plot(q_vals)
    plt.title('D3QN Convergence (Compensated Subgroup)')
    plt.ylabel('Mean Q-Value (V)')
    plt.xlabel('Training Steps (Epochs)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(BASE_DIR, f"q_convergence_{TIMESTAMP}.png"))
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
    plt.savefig(os.path.join(BASE_DIR, f"ope_mortality_curve_{TIMESTAMP}.png"))
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

# 2026-04-02 01:55:51,429 - INFO - ========================================
# 2026-04-02 01:55:51,429 - INFO - Starting D3QN Pipeline Execution
# 2026-04-02 01:55:51,429 - INFO - ========================================
# 2026-04-02 01:55:51,429 - INFO - Connecting to Database and Loading Data...
# 2026-04-02 01:55:51,960 - INFO - Filtering complete. Found 192246 patient records.
# 2026-04-02 01:55:51,960 - INFO - Mapping 5 Vasopressor Actions...
# 2026-04-02 01:55:52,055 - INFO - Calculating State-Based Rewards (Physiological Distress)...
# 2026-04-02 01:55:52,211 - INFO - Normalizing Features...
# 2026-04-02 01:55:52,426 - INFO - Initializing D3QN on cuda...
# 2026-04-02 01:55:52,601 - INFO - Loading data into ReplayBuffer...
# 2026-04-02 01:56:02,022 - INFO - Starting Training Epochs...
# 2026-04-02 01:56:03,325 - INFO - Epoch  100 | Loss: 0.9957 | Q-Avg: 0.2704 | Eps: 0.80
# 2026-04-02 01:56:04,255 - INFO - Epoch  200 | Loss: 0.9480 | Q-Avg: 0.2686 | Eps: 0.60
# 2026-04-02 01:56:05,188 - INFO - Epoch  300 | Loss: 0.8206 | Q-Avg: 0.2479 | Eps: 0.40
# 2026-04-02 01:56:06,118 - INFO - Epoch  400 | Loss: 0.9070 | Q-Avg: 0.2677 | Eps: 0.20
# 2026-04-02 01:56:07,047 - INFO - Epoch  500 | Loss: 0.8975 | Q-Avg: 0.2378 | Eps: 0.05
# 2026-04-02 01:56:07,979 - INFO - Epoch  600 | Loss: 0.6721 | Q-Avg: 0.3178 | Eps: 0.05
# 2026-04-02 01:56:08,946 - INFO - Epoch  700 | Loss: 0.5510 | Q-Avg: 0.2321 | Eps: 0.05
# 2026-04-02 01:56:09,881 - INFO - Epoch  800 | Loss: 0.6072 | Q-Avg: 0.3368 | Eps: 0.05
# 2026-04-02 01:56:10,808 - INFO - Epoch  900 | Loss: 0.5335 | Q-Avg: 0.2581 | Eps: 0.05
# 2026-04-02 01:56:11,749 - INFO - Epoch 1000 | Loss: 0.4043 | Q-Avg: 0.3361 | Eps: 0.05
# 2026-04-02 01:56:11,750 - INFO - Running Off-Policy Evaluation...
# 2026-04-02 01:56:11,808 - INFO - Generating Plots...
# 2026-04-02 01:56:12,227 - INFO - All Plots Saved to /home/howard900126 with tag 0402_0155
# 2026-04-02 01:56:12,227 - INFO - Pipeline completed successfully!