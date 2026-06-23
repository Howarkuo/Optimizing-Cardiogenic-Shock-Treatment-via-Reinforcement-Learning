# Tables Used (Input): 

# rl_state_space: The primary source for hourly clinical features including blood pressure (SBP, MAP), treatments (vaso rate, MCS status), and laboratory values (lactate, creatinine, pH, urine output).
# cohort: Used to retrieve the hospital_expire_flag (mortality label) for each patient stay.
# bsa_ci_processed: Queried to obtain the static weight_kg for each stay_id, ensuring correct weight-based urine output calculations.

# Tables Created (Internal & Exported)
# df (Internal Pandas DataFrame)
# patient_flags
# Phenotype ID Lists (CSV Files)

import duckdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Fix Pandas warning for future downcasting behavior
pd.set_option('future.no_silent_downcasting', True)
# ==========================================
# ⚙️ CONFIGURATION & THRESHOLDS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mimic_shock.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "phenotype_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Thresholds (Updated to User Specs) ---
TH_LACTATE = 2.0        # mmol/L (> 2.0)
TH_URINE_ABS = 30       # mL/hr (< 30)
TH_URINE_KG = 0.5       # ml/kg/hr (< 0.5)
TH_PH_SEVERE = 7.2      # (< 7.2 severe acidosis)
TH_CREATININE = 2.0     # mg/dL (Approx 2x Upper Limit of Normal)
TH_SBP_ABS = 90         # mmHg (< 90)
TH_MAP_ABS = 65         # mmHg (< 65)
TH_SBP_DROP = 30        # mmHg (Drop >= 30 from baseline)

def run_phenotyping():
    print(f"🔬 [Step 5] Running Clinical Phenotype Analysis (Strict Criteria)...")
    
    if not os.path.exists(DB_PATH):
        print("❌ Error: DB not found.")
        return

    con = duckdb.connect(DB_PATH)
    
    # 1. Load State Space + Weight + Mortality + MCS
    print("   -> Loading data from DuckDB...")
    df = con.execute("""
        SELECT 
            r.stay_id, 
            r.chart_hour, 
            r.sbp, 
            r.map,
            r.vaso_rate, 
            r.mcs_active,
            r.lactate, 
            r.urine_output, 
            r.ph, 
            r.creatinine, 
            c.hospital_expire_flag as mortality,
            
            -- Get static weight (max per stay to avoid duplicates)
            b.weight_kg
            
        FROM rl_state_space r
        JOIN cohort c ON r.stay_id = c.stay_id
        LEFT JOIN (
            SELECT stay_id, MAX(weight_kg) as weight_kg 
            FROM bsa_ci_processed 
            GROUP BY stay_id
        ) b ON r.stay_id = b.stay_id
    """).fetchdf()
    
    con.close()
    print(f"      Loaded {len(df):,} hourly records.")

    # ==============================================================================
    # 2. CALCULATE BASELINE SBP
    # ==============================================================================
    print("   -> Calculating Baseline SBP...")
    # Sort by time and take first SBP per stay_id
    df = df.sort_values(['stay_id', 'chart_hour'])
    baseline_sbp = df.groupby('stay_id')['sbp'].transform('first')
    df['baseline_sbp'] = baseline_sbp

    # ==============================================================================
    # 3. APPLY PHENOTYPE LOGIC
    # ==============================================================================
    print("   -> Applying T_perf and T_hypo criteria...")
    
    # Fill weight with 70kg default if missing to prevent division error
    df['weight_kg'] = df['weight_kg'].fillna(70)
    df['urine_ml_kg_hr'] = df['urine_output'] / df['weight_kg']

    # --- T_hypo: Hypotension ---
    # incorrect: this is only for any dip within an hour only
    # # Rule 1: SBP < 90 AND MAP < 65 (Simulates sustained severe hypo)
    # cond_hypo_abs = (df['sbp'] < TH_SBP_ABS) & (df['map'] < TH_MAP_ABS)
    # --- T_hypo: Hypotension ---
    
    # 1. Identify 'Instant' hypotension (any dip within an hour)
    hypo_instant = (df['sbp'] < TH_SBP_ABS) & (df['map'] < TH_MAP_ABS)

    # Add it to the dataframe temporarily so groupby can use it
    df['temp_hypo_instant'] = hypo_instant

    # 2. Apply 'Sustained' logic (True only if present for 2 consecutive hours)
    # This ensures the condition lasted for at least 60 minutes total.
    cond_hypo_sustained = df['temp_hypo_instant'] & df.groupby('stay_id')['temp_hypo_instant'].shift(1).fillna(False)

    # Rule 2: Drop in SBP >= 30 from baseline
    cond_sbp_drop = df['sbp'] <= (df['baseline_sbp'] - TH_SBP_DROP)
    
    # Rule 3: Support Needed (Vasopressors OR MCS)
    cond_support = (df['vaso_rate'] > 0) | (df['mcs_active'] > 0)
    
    # Combine (Using cond_hypo_sustained instead of the deleted cond_hypo_abs)
    df['T_hypo'] = cond_hypo_sustained | cond_sbp_drop | cond_support
    
    # Clean up the temporary column
    df = df.drop(columns=['temp_hypo_instant'])

    # --- T_perf: Hypoperfusion ---
    # Rule 1: Elevated Lactate
    cond_lactate = df['lactate'] > TH_LACTATE
    
    # Rule 2: Low Urine (< 30ml/hr OR < 0.5 ml/kg/hr)
    cond_urine = (df['urine_output'] < TH_URINE_ABS) | (df['urine_ml_kg_hr'] < TH_URINE_KG)
    
    # Rule 3: Organ Dysfunction (Cr >= 2.0 OR pH < 7.2)
    cond_organ = (df['creatinine'] >= TH_CREATININE) | (df['ph'] < TH_PH_SEVERE)
    
    # T_perf = Lactate AND Urine AND OrganFailure
    df['T_perf'] = cond_lactate & cond_urine & cond_organ

    # ==============================================================================
    # 4. DETERMINE PATIENT PHENOTYPE
    # ==============================================================================
    print("   -> Aggregating to Patient Level...")

    # Determine if conditions were met simultaneously (Classic Shock)
    df['is_classic'] = df['T_perf'] & df['T_hypo']
    
    # Aggregating max flags per patient
    patient_flags = df.groupby('stay_id').agg({
        'T_perf': 'max',
        'T_hypo': 'max',
        'is_classic': 'max',
        'mortality': 'max'
    }).reset_index()
    def classify(row):
        # Hierarchy: Classic > Normotensive > Compensated > Neither
        if row['is_classic']:
            return "Classic Shock"
        elif row['T_perf']:
            return "Normotensive Cardiogenic Shock"
        elif row['T_hypo']:
            return "Compensated Shock"
        else:
            return "Neither"

    patient_flags['Phenotype'] = patient_flags.apply(classify, axis=1)

    # def classify(row):
    #     # Hierarchy: Classic > Cryptic > Preserved > Neither
    #     if row['is_classic']:
    #         return "Both (Classic Shock)"
    #     elif row['T_perf']:
    #         return "Perf Only (Cryptic Shock)"
    #     elif row['T_hypo']:
    #         return "Hypo Only (Preserved Perfusion)"
    #     else:
    #         return "Neither"

    # patient_flags['Phenotype'] = patient_flags.apply(classify, axis=1)

    # ==============================================================================
    # 5. GENERATE PLOT & EXPORT
    # ==============================================================================
    print("   -> Generating Visualization...")
    summary = patient_flags.groupby(['mortality', 'Phenotype']).size().reset_index(name='count')
    totals = summary.groupby('mortality')['count'].transform('sum')
    summary['percent'] = (summary['count'] / totals) * 100
    summary['Group'] = summary['mortality'].map({0: 'Survivor', 1: 'Non-Survivor'})
    
    pivot_df = summary.pivot(index='Group', columns='Phenotype', values='percent').fillna(0)
    
    # --- FIX: Set Specific Order for Rows ---
    pivot_df = pivot_df.reindex(['Survivor', 'Non-Survivor'])
    
    # Set Specific Order for Columns
    # Set Specific Order for Columns to match your new manuscript
    order = ['Classic Shock', 'Compensated Shock', 'Normotensive Cardiogenic Shock', 'Neither']
    pivot_df = pivot_df.reindex(columns=order).fillna(0) 

    # Plotting
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 7), 
                        color=['#d62728', '#ff7f0e', '#2ca02c', '#7f7f7f']) 

    plt.title("Phenotype Distribution by Survival Status", fontsize=14, fontweight='bold')
    plt.ylabel("Percentage of Patients (%)", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=0, fontsize=12)
    
    # Move legend outside
    plt.legend(title='Clinical Phenotype', bbox_to_anchor=(1.05, 1), loc='upper left')

    for c in ax.containers:
        # Only label if segment is big enough to read
        ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', fontweight='bold', padding=0)

    plt.tight_layout()
    
    # Save chart and fix the mismatched print statement
    save_path = "phenotype_distribution-new.png"
    plt.savefig(os.path.join(OUTPUT_DIR, save_path), dpi=300)
    print(f"      Chart saved to: {OUTPUT_DIR}/{save_path}")
    
    # Export CSVs
    print("   -> Exporting ID lists...")
    for pheno in order:
        ids = patient_flags[patient_flags['Phenotype'] == pheno]['stay_id']
        # This will safely extract the first word (e.g., "Normotensive" or "Compensated")
        fname = pheno.split(' ')[0] + "_ids.csv" 
        ids.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
        print(f"      Saved {len(ids)} IDs for {pheno} (as {fname})")

if __name__ == "__main__":
    run_phenotyping()
#     order = ['Both (Classic Shock)', 'Hypo Only (Preserved Perfusion)', 'Perf Only (Cryptic Shock)', 'Neither']
#     pivot_df = pivot_df.reindex(columns=order).fillna(0) 

#     # Plotting
#     ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 7), 
#                         color=['#d62728', '#ff7f0e', '#2ca02c', '#7f7f7f']) 

#     plt.title("Phenotype Distribution by Survival Status", fontsize=14, fontweight='bold')
#     plt.ylabel("Percentage of Patients (%)", fontsize=12)
#     plt.xlabel("")
#     plt.xticks(rotation=0, fontsize=12)
    
#     # Move legend outside
#     plt.legend(title='Clinical Phenotype', bbox_to_anchor=(1.05, 1), loc='upper left')

#     for c in ax.containers:
#         # Only label if segment is big enough to read
#         ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', fontweight='bold', padding=0)

#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, "phenotype_distribution-new.png"), dpi=300)
#     print(f"      Chart saved to: {OUTPUT_DIR}/phenotype_distribution.png")
    
#     # Export CSVs
#     print("   -> Exporting ID lists...")
#     for pheno in order:
#         ids = patient_flags[patient_flags['Phenotype'] == pheno]['stay_id']
#         fname = pheno.split(' ')[0] + "_ids.csv"
#         ids.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
#         print(f"      Saved {len(ids)} IDs for {pheno}")

# if __name__ == "__main__":
#     run_phenotyping()

#  .\step 005_Phenotype_Analysis_new.py
# 🔬 [Step 5] Running Clinical Phenotype Analysis (Strict Criteria)...
#    -> Loading data from DuckDB...
#       Loaded 420,366 hourly records.
#    -> Calculating Baseline SBP...
#    -> Applying T_perf and T_hypo criteria...
# C:\Users\howar\Desktop\DHLAB_code\cardiogenic_shock\0128_NEW\005_Phenotype_Analysis_new.py:109: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
#   cond_hypo_sustained = df['temp_hypo_instant'] & df.groupby('stay_id')['temp_hypo_instant'].shift(1).fillna(False)
#    -> Aggregating to Patient Level...
#    -> Generating Visualization...
#       Chart saved to: C:\Users\howar\Desktop\DHLAB_code\cardiogenic_shock\0128_NEW\phenotype_results/phenotype_distribution.png
#    -> Exporting ID lists...
#       Saved 736 IDs for Both (Classic Shock)
#       Saved 1160 IDs for Hypo Only (Preserved Perfusion)
#       Saved 177 IDs for Perf Only (Cryptic Shock)
#       Saved 298 IDs for Neither 