import duckdb
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import traceback
from tqdm import trange  # <-- Added tqdm here
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tianshou.data import Batch, ReplayBuffer
from tianshou.utils.net.common import Net
from tianshou.policy import DQNPolicy

# ==========================================
# ⚙️ 1. CONFIGURATION & GPU SETUP
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
# 📊 2. DATA PREPROCESSING & ACTION MAPPING
# ==========================================
def prepare_clinical_data():
    logging.info(" Connecting to Database and Loading Data...")
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
    
    logging.info(f" Filtering complete. Found {len(df)} patient records.")

    # --- ACTION MAPPING ---
    logging.info(" Mapping 5 Vasopressor Actions...")
    active_vaso = df[df['vaso_rate'] > 0]['vaso_rate']
    vaso_bins = [0] + list(active_vaso.quantile([0.25, 0.5, 0.75, 1.0]))
    
    def get_action(val):
        if val <= 0: return 0
        for i in range(len(vaso_bins)-1):
            if val <= vaso_bins[i+1]: return i + 1
        return 4
    df['action'] = df['vaso_rate'].apply(get_action)

    # --- REWARD CALCULATION ---
    logging.info(" Calculating Rewards...")
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
    
    # df[FEATURES] = scaler.transform(df[FEATURES])
    # next_features = [f'next_{c}' for c in FEATURES]
    # df[next_features] = scaler.transform(df[next_features])
    # Scale current and next features
    df[FEATURES] = scaler.transform(df[FEATURES])
    next_features = [f'next_{c}' for c in FEATURES]
    
    # Add .values to bypass the strict column name check
    df[next_features] = scaler.transform(df[next_features].values)
    
    con.close()
    return df, train_ids, val_ids

# # ==========================================
# # 🤖 3. D3QN ARCHITECTURE & TRAINING
# # ==========================================
# def train_d3qn(df, train_ids):
#     logging.info(f"🤖 Initializing D3QN on {DEVICE}...")
#     train_df = df[df['stay_id'].isin(train_ids)].copy()
    
#     buffer = ReplayBuffer(size=len(train_df))
#     next_features = [f'next_{c}' for c in FEATURES]
    
#     buffer.add(Batch(
#         obs=train_df[FEATURES].values,
#         act=train_df['action'].values,
#         rew=train_df['reward'].values,
#         obs_next=train_df[next_features].values,
#         terminated=train_df['done'].values.astype(bool),
#         truncated=np.zeros(len(train_df), dtype=bool)
#     ))

#     net = Net(state_shape=len(FEATURES), action_shape=5, hidden_sizes=[256, 256], device=DEVICE,
#               dueling_param=({"hidden_sizes": [128]}, {"hidden_sizes": [128]})).to(DEVICE)
    
#     policy = DQNPolicy(model=net, optim=torch.optim.Adam(net.parameters(), lr=LR),
#                        discount_factor=GAMMA, estimation_step=3, target_update_freq=500, is_double=True)

#     q_vals, losses = [], []

#     logging.info("🚀 Starting Training Epochs...")
    
#     # --- TQDM PROGRESS BAR ADDED HERE ---
#     pbar = trange(1, EPISODES + 1, desc="Training D3QN", unit="ep")
    
#     for epoch in pbar:
#         stats = policy.update(sample_size=BATCH_SIZE, buffer=buffer)
        
#         current_q = stats.get('v_avg', 0)
#         current_loss = stats.get('loss', 0)
        
#         q_vals.append(current_q)
#         losses.append(current_loss)
        
#         eps = max(EPS_MIN, 1.0 - (epoch / 500))
#         policy.set_eps(eps)
        
#         # Update the progress bar dynamically with live metrics
#         pbar.set_postfix(loss=f"{current_loss:.4f}", q_avg=f"{current_q:.4f}", eps=f"{eps:.2f}")
        
#         # Keep permanent log checkpoints every 100 epochs (won't disrupt the bar significantly)
#         if epoch % 100 == 0:
#             logging.info(f"Epoch {epoch:4d} | Loss: {current_loss:.4f} | Q-Avg: {current_q:.4f} | Eps: {eps:.2f}")

#     return policy, q_vals, losses

# ==========================================
# 🤖 3. D3QN ARCHITECTURE & TRAINING
# ==========================================
def train_d3qn(df, train_ids):
    logging.info(f" Initializing D3QN on {DEVICE}...")
    train_df = df[df['stay_id'].isin(train_ids)].copy()
    
    buffer = ReplayBuffer(size=len(train_df))
    next_features = [f'next_{c}' for c in FEATURES]
    
    # 1. Create the Batch from your dataframe
    dataset_batch = Batch(
        obs=train_df[FEATURES].values,
        act=train_df['action'].values,
        rew=train_df['reward'].values,
        obs_next=train_df[next_features].values,
        terminated=train_df['done'].values.astype(bool),
        truncated=np.zeros(len(train_df), dtype=bool)
    )

    # 2. Add transitions one by one
    logging.info(" Loading data into ReplayBuffer...")
    for i in trange(len(train_df), desc="Filling Buffer", unit="step"):
        buffer.add(dataset_batch[i])

    net = Net(state_shape=len(FEATURES), action_shape=5, hidden_sizes=[256, 256], device=DEVICE,
              dueling_param=({"hidden_sizes": [128]}, {"hidden_sizes": [128]})).to(DEVICE)
    
    policy = DQNPolicy(model=net, optim=torch.optim.Adam(net.parameters(), lr=LR),
                       discount_factor=GAMMA, estimation_step=3, target_update_freq=500, is_double=True)

    q_vals, losses = [], []

    logging.info(" Starting Training Epochs...")
    
    # --- TQDM PROGRESS BAR ADDED HERE ---
    pbar = trange(1, EPISODES + 1, desc="Training D3QN", unit="ep")
    
    # for epoch in pbar:
    #     stats = policy.update(sample_size=BATCH_SIZE, buffer=buffer)
        
    #     current_q = stats.get('v_avg', 0)
    #     current_loss = stats.get('loss', 0)
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
# 📉 4. EVALUATION & PLOTTING (OPE)
# ==========================================
def evaluate_and_plot(policy, df, val_ids, q_vals):
    logging.info(" Running Off-Policy Evaluation...")
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
    
    logging.info(" Generating Plots...")
    plt.figure(figsize=(10, 5))
    plt.plot(q_vals)
    plt.title('D3QN Convergence (Compensated Subgroup)')
    plt.ylabel('Mean Q-Value (V) / Loss Tracker')
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
    
    logging.info(f" All Plots Saved to {BASE_DIR}")

# ==========================================
# 🚀 5. MAIN EXECUTION WITH ERROR CATCHING
# ==========================================
if __name__ == "__main__":
    try:
        logging.info("========================================")
        logging.info(" Starting D3QN Pipeline Execution")
        logging.info("========================================")
        
        full_df, train_idx, val_idx = prepare_clinical_data()
        model_policy, q_history, loss_history = train_d3qn(full_df, train_idx)
        evaluate_and_plot(model_policy, full_df, val_idx, q_history)
        
        logging.info(" Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(" A fatal error occurred during execution:", exc_info=True)