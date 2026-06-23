# import duckdb
# import os
# import pandas as pd
# import numpy as np
# import torch
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # ==========================================
# # ⚙️ CONFIGURATION
# # ==========================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB_PATH = os.path.join(BASE_DIR, "mimic_shock.db")
# PHENO_DIR = os.path.join(BASE_DIR, "phenotype_results")

# # Features to be used as State Input (S_t)
# FEATURES = [
#     'sbp', 'map', 'hr', 'resp', 'temp', 'spo2', 'cvp', 'pcwp', 
#     'lactate', 'creatinine', 'glucose', 'potassium', 'sodium', 
#     'ph', 'po2', 'pco2', 'fio2', 'urine_output', 'vaso_rate', 'age'
# ]

# def generate_rl_tensors():
#     con = duckdb.connect(DB_PATH)
    
#     # 1. Identify "Vaso-Users" (Stay IDs where max(vaso_rate) > 0)
#     print("🔍 [1/5] Identifying patients with active vasopressor usage...")
#     vaso_user_query = """
#         SELECT stay_id 
#         FROM rl_state_space 
#         GROUP BY stay_id 
#         HAVING MAX(vaso_rate) > 0
#     """
#     vaso_user_ids = con.execute(vaso_user_query).fetchdf()['stay_id'].tolist()
    
#     # 2. Load and Filter Data
#     print(f"📦 [2/5] Loading data for {len(vaso_user_ids)} vasopressor users...")
    
#     # Load hourly states
#     df = con.execute("SELECT * FROM rl_state_space").fetchdf()
#     df = df[df['stay_id'].isin(vaso_user_ids)].sort_values(['stay_id', 'chart_hour'])
    
#     # Load Phenotypes & Mortality for stratification
#     # (Assuming you have a mapping of stay_id -> Phenotype and mortality)
#     df_labels = con.execute("""
#         SELECT stay_id, hospital_expire_flag as mortality 
#         FROM cohort
#     """).fetchdf()
    
#     # Re-merge Phenotypes (using your logic from Step 6b)
#     # Note: Ensure your local files are up to date
#     df = df.merge(df_labels, on='stay_id', how='left')

#     # 3. Stratified 80/20 Split by Stay ID
#     print("⚖️ [3/5] Performing Stratified Split (80/20)...")
#     unique_stays = df[['stay_id', 'mortality']].drop_duplicates()
    
#     train_ids, val_ids = train_test_split(
#         unique_stays['stay_id'], 
#         test_size=0.20, 
#         stratify=unique_stays['mortality'], 
#         random_state=42
#     )

#     # 4. Normalization (Input Scaling)
#     # Neural networks converge faster when inputs are normalized:
#     # z = (x - mean) / std
#     print("📈 [4/5] Normalizing features (StandardScaler fit on Train only)...")
    
#     # Log-transform skewed variables (Lactate, Creatinine, Vaso_rate)
#     skewed_cols = ['lactate', 'creatinine', 'vaso_rate', 'urine_output']
#     for col in skewed_cols:
#         df[col] = np.log1p(df[col]) # ln(x + 1)
    
#     scaler = StandardScaler()
    
#     # Fit ONLY on training data to prevent data leakage
#     scaler.fit(df[df['stay_id'].isin(train_ids)][FEATURES])
    
#     # Transform both
#     df[FEATURES] = scaler.transform(df[FEATURES])

#     # 5. Convert to Tensors
#     print("🤖 [5/5] Generating Tensors for D3QN...")
    
#     X_train = torch.tensor(df[df['stay_id'].isin(train_ids)][FEATURES].values, dtype=torch.float32)
#     X_val = torch.tensor(df[df['stay_id'].isin(val_ids)][FEATURES].values, dtype=torch.float32)

#     print("-" * 40)
#     print(f"✅ Success! Data ready for D3QN training.")
#     print(f"   Train Tensors: {X_train.shape}")
#     print(f"   Val Tensors:   {X_val.shape}")
#     print("-" * 40)
    
#     # Optional: Save for remote server
#     # torch.save(X_train, 'X_train.pt')
#     # torch.save(X_val, 'X_val.pt')
    
#     con.close()
#     return X_train, X_val

# if __name__ == "__main__":
#     X_train, X_val = generate_rl_tensors()

# # PS C:\Users\howar\Desktop\DHLAB_code\cardiogenic_shock\0128_NEW> poetry run python .\007_tensor_creation.py      
# # 🔍 [1/5] Identifying patients with active vasopressor usage...
# # 📦 [2/5] Loading data for 1778 vasopressor users...
# # ⚖️ [3/5] Performing Stratified Split (80/20)...
# # 📈 [4/5] Normalizing features (StandardScaler fit on Train only)...
# # 🤖 [5/5] Generating Tensors for D3QN...
# # ----------------------------------------
# # ✅ Success! Data ready for D3QN training.
# #    Train Tensors: torch.Size([287768, 20])
# #    Val Tensors:   torch.Size([73876, 20])
# # ----------------------------------------


# let's focus on the Compensated subgroup first (survivors have a median max NED of 0.10)
import duckdb
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mimic_shock.db")
PHENO_DIR = os.path.join(BASE_DIR, "phenotype_results") # Folder from Step 5

FEATURES = [
    'sbp', 'map', 'hr', 'resp', 'temp', 'spo2', 'cvp', 'pcwp', 
    'lactate', 'creatinine', 'glucose', 'potassium', 'sodium', 
    'ph', 'po2', 'pco2', 'fio2', 'urine_output', 'vaso_rate', 'age'
]

def generate_rl_tensors():
    con = duckdb.connect(DB_PATH)
    
    # 1. Identify "Vaso-Users"
    print("🔍 [1/5] Identifying patients with active vasopressor usage...")
    vaso_user_ids = con.execute("""
        SELECT stay_id FROM rl_state_space 
        GROUP BY stay_id HAVING MAX(vaso_rate) > 0
    """).fetchdf()['stay_id'].tolist()
    
    # 2. Identify "Compensated Shock" patients (NEW SUBGROUP FILTER)
    print("🎯 [1b/5] Filtering for Compensated Shock subgroup...")
    comp_file = os.path.join(PHENO_DIR, "Compensated_ids.csv") # Check your filename matches!
    if not os.path.exists(comp_file):
        print(f"❌ Error: {comp_file} not found. Run Step 5 first.")
        return
    
    comp_ids = pd.read_csv(comp_file)['stay_id'].tolist()
    
    # INTERSECTION: Must be in Compensated Group AND use Vasopressors
    target_ids = list(set(vaso_user_ids) & set(comp_ids))
    print(f"📊 Found {len(target_ids)} Compensated Shock patients who use vasopressors.")

    # 3. Load and Filter Data
    df = con.execute("SELECT * FROM rl_state_space").fetchdf()
    df = df[df['stay_id'].isin(target_ids)].sort_values(['stay_id', 'chart_hour'])
    
    # Load Mortality for stratification
    df_labels = con.execute("SELECT stay_id, hospital_expire_flag as mortality FROM cohort").fetchdf()
    df = df.merge(df_labels, on='stay_id', how='left')

    # 4. Stratified 80/20 Split
    unique_stays = df[['stay_id', 'mortality']].drop_duplicates()
    train_ids, val_ids = train_test_split(
        unique_stays['stay_id'], 
        test_size=0.20, 
        stratify=unique_stays['mortality'], 
        random_state=42
    )

    # 5. Normalization (Paper-Style: Log -> StandardScaler)
    print("📈 [4/5] Normalizing features (Log-transform for skewed cols)...")
    skewed_cols = ['lactate', 'creatinine', 'vaso_rate', 'urine_output']
    for col in skewed_cols:
        df[col] = np.log1p(df[col])
    
    scaler = StandardScaler()
    # Fit only on the Compensated training subset!
    scaler.fit(df[df['stay_id'].isin(train_ids)][FEATURES])
    df[FEATURES] = scaler.transform(df[FEATURES])

    # 6. Generate Tensors
    X_train = torch.tensor(df[df['stay_id'].isin(train_ids)][FEATURES].values, dtype=torch.float32)
    X_val = torch.tensor(df[df['stay_id'].isin(val_ids)][FEATURES].values, dtype=torch.float32)

    print("-" * 40)
    print(f"✅ Success! Compensated Shock Subgroup Ready.")
    print(f"   Train Tensors: {X_train.shape}")
    print(f"   Val Tensors:   {X_val.shape}")
    print("-" * 40)
    
    con.close()
    return X_train, X_val

if __name__ == "__main__":
    generate_rl_tensors()


#  .\007_tensor_creation.py      
# 🔍 [1/5] Identifying patients with active vasopressor usage...
# 🎯 [1b/5] Filtering for Compensated Shock subgroup...
# 📊 Found 964 Compensated Shock patients who use vasopressors.
# 📈 [4/5] Normalizing features (Log-transform for skewed cols)...
# ----------------------------------------
# ✅ Success! Compensated Shock Subgroup Ready.
#    Train Tensors: torch.Size([137399, 20])
#    Val Tensors:   torch.Size([32489, 20])
# ----------------------------------------

