import duckdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mimic_shock.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Phenotype Thresholds (Same as Step 5)
TH_LACTATE = 2.0
TH_URINE_KG = 0.5
TH_PH_SEVERE = 7.2
TH_CREATININE = 2.0
TH_SBP_DROP = 30
TH_SBP_ABS = 90
TH_MAP_ABS = 65

def run_endpoint_analysis():
    print(f"📈 [Step 6] Running Study Endpoints & Subgroup Analysis...")
    
    if not os.path.exists(DB_PATH):
        print("❌ Error: DB not found.")
        return

    con = duckdb.connect(DB_PATH)

    # ==============================================================================
    # 1. LOAD DATA & RE-CLASSIFY PHENOTYPES
    # ==============================================================================
    print("   -> Loading Cohort & State Space...")
    
    # Load Hourly State for Phenotyping & Organ Failure Calc
    df_state = con.execute("""
        SELECT 
            r.stay_id, r.chart_hour, r.sbp, r.map, r.vaso_rate, r.mcs_active,
            r.lactate, r.urine_output, r.ph, r.creatinine, r.po2, r.fio2,
            b.weight_kg
        FROM rl_state_space r
        LEFT JOIN (SELECT stay_id, MAX(weight_kg) as weight_kg FROM bsa_ci_processed GROUP BY stay_id) b 
            ON r.stay_id = b.stay_id
    """).fetchdf()
    
    # Load Cohort Outcomes
    df_cohort = con.execute("""
        SELECT 
            stay_id, subject_id, hadm_id, 
            intime, outtime, 
            dod, hospital_expire_flag
        FROM cohort
    """).fetchdf()

    # --- Apply Phenotype Logic (Simplified from Step 5) ---
    print("   -> Classifying Phenotypes...")
    df_state['weight_kg'] = df_state['weight_kg'].fillna(70)
    df_state['urine_ml_kg'] = df_state['urine_output'] / df_state['weight_kg']
    
    # Baseline SBP
    df_state = df_state.sort_values(['stay_id', 'chart_hour'])
    df_state['base_sbp'] = df_state.groupby('stay_id')['sbp'].transform('first')
    
    # Conditions
    # cond_hypo = (
    #     ((df_state['sbp'] < TH_SBP_ABS) & (df_state['map'] < TH_MAP_ABS)) |
    #     (df_state['sbp'] <= (df_state['base_sbp'] - TH_SBP_DROP)) |
    #     (df_state['vaso_rate'] > 0) | (df_state['mcs_active'] > 0)
    # )
    # --- Update Step 6 with Sustained Logic ---
    df_state['hypo_instant'] = (df_state['sbp'] < TH_SBP_ABS) & (df_state['map'] < TH_MAP_ABS)

    # Apply Sustained (2-hour) logic
    cond_hypo_sustained = (
        df_state['hypo_instant'] & 
        df_state.groupby('stay_id')['hypo_instant'].shift(1).fillna(False)
    )

    # Final Hypo condition including Vasopressors/MCS
    cond_hypo = (
        cond_hypo_sustained |
        (df_state['sbp'] <= (df_state['base_sbp'] - TH_SBP_DROP)) |
        (df_state['vaso_rate'] > 0) | (df_state['mcs_active'] > 0)
    )
    cond_perf = (
        (df_state['lactate'] > TH_LACTATE) & 
        ((df_state['urine_output'] < 30) | (df_state['urine_ml_kg'] < TH_URINE_KG)) & 
        ((df_state['creatinine'] >= TH_CREATININE) | (df_state['ph'] < TH_PH_SEVERE))
    )
    
    # Aggregating to Patient
    df_state['T_hypo'] = cond_hypo
    df_state['T_perf'] = cond_perf
    df_state['is_classic'] = cond_hypo & cond_perf
    
    phenos = df_state.groupby('stay_id')[['T_hypo', 'T_perf', 'is_classic']].max().reset_index()
    
    def get_pheno(row):
        if row['is_classic']: return "Classic Shock"
        if row['T_perf']: return "Normotensive Cardiogenic Shock"
        if row['T_hypo']: return " Compensated Shock "
        return "Neither"
    
    phenos['Phenotype'] = phenos.apply(get_pheno, axis=1)
    
    # Merge Phenotype into Cohort
    df_final = df_cohort.merge(phenos[['stay_id', 'Phenotype']], on='stay_id', how='left')

    # ==============================================================================
    # 2. EXTRACT ADDITIONAL EVENTS (RRT, Transplant, Cardiac Arrest)
    # ==============================================================================
    print("   -> Extracting Adverse Events (RRT, Transplant, Arrest)...")

    # RRT (Dialysis) from ProcedureEvents
    rrt_ids = "225441, 225802, 225803, 225805, 225809"
    df_rrt = con.execute(f"""
        SELECT stay_id, 1 as rrt_flag 
        FROM procedureevents 
        WHERE itemid IN ({rrt_ids})
        GROUP BY stay_id
    """).fetchdf()
    
    # Cardiac Arrest & Transplant from Diagnoses (ICD)
    df_icd = con.execute("""
        SELECT 
            hadm_id,
            MAX(CASE WHEN icd_code LIKE 'I46%' OR icd_code = '4275' THEN 1 ELSE 0 END) as arrest_flag,
            MAX(CASE WHEN icd_code = 'Z941' OR icd_code LIKE '02HA0%' THEN 1 ELSE 0 END) as transplant_flag
        FROM diagnoses_icd
        GROUP BY hadm_id
    """).fetchdf()

    # MCS from State Space (already have flag)
    mcs_flags = df_state.groupby('stay_id')['mcs_active'].max().reset_index().rename(columns={'mcs_active': 'mcs_flag'})

    # Merge all events
    df_final = df_final.merge(df_rrt, on='stay_id', how='left').fillna({'rrt_flag': 0})
    df_final = df_final.merge(df_icd, on='hadm_id', how='left').fillna({'arrest_flag': 0, 'transplant_flag': 0})
    df_final = df_final.merge(mcs_flags, on='stay_id', how='left')
        # ==============================================================================
    # 3. CALCULATE MORTALITY & ORGAN-FREE DAYS (REVISED)
    # ==============================================================================
    print("   -> Calculating Mortality & Corrected Survival Times...")

    # Ensure datetime conversion
    df_final['intime'] = pd.to_datetime(df_final['intime'])
    df_final['outtime'] = pd.to_datetime(df_final['outtime'])
    df_final['dod'] = pd.to_datetime(df_final['dod'])

    # 1. Determine Time to Event or Censoring
    # Use DOD if available, otherwise use OUTTIME as a proxy for death/last contact
    df_final['end_of_followup'] = df_final['dod'].fillna(df_final['outtime'])

    # 2. Calculate duration in days
    df_final['duration'] = (df_final['end_of_followup'] - df_final['intime']).dt.total_seconds() / 86400

    # 3. Handle Negative/Zero Times (Early death on day of admission)
    # We set a minimum of 0.1 days (2.4 hours) so the KM Fitter can plot them
    df_final['duration'] = df_final['duration'].clip(lower=0.1)

    # 4. Define Event (E) and Time (T) for the 90-day window
    # T = time to death or 90 days (whichever is first)
    df_final['T_90'] = df_final['duration'].clip(upper=90)
    # E = did they die within 90 days?
    df_final['E_90'] = np.where((df_final['hospital_expire_flag'] == 1) & (df_final['duration'] <= 90), 1, 0)

    # Debug: Check for dropped rows
    dropped = df_final[df_final['T_90'].isna() | (df_final['T_90'] <= 0)]
    print(f"      [Debug] Patients with invalid times: {len(dropped)}")

    # ==============================================================================
    # 3. CALCULATE MORTALITY & ORGAN-FREE DAYS (Fixed for Negative Durations)
    # ==============================================================================
    print("   -> Calculating Mortality & Organ-Free Days...")
    
    # Convert dates
    df_final['intime'] = pd.to_datetime(df_final['intime'])
    df_final['dod'] = pd.to_datetime(df_final['dod'])
    
    # Days to Death
    # FIX: We clip negative values to 0.001 (approx 1 min) so lifelines doesn't drop them
    df_final['days_to_death'] = (df_final['dod'] - df_final['intime']).dt.total_seconds() / 86400
    df_final['days_to_death'] = df_final['days_to_death'].clip(lower=0.001)
    
    # 30-day and 90-day Mortality Flags
    df_final['mortality_30d'] = np.where(df_final['days_to_death'].notnull() & (df_final['days_to_death'] <= 30), 1, 0)
    df_final['mortality_90d'] = np.where(df_final['days_to_death'].notnull() & (df_final['days_to_death'] <= 90), 1, 0)
    
    # Organ-Free Days Calculation (Simplified SOFA approach)
    df_state['pf_ratio'] = df_state['po2'] / (df_state['fio2'] / 100.0)
    
    df_state['fail_cv'] = (df_state['vaso_rate'] > 0) | (df_state['map'] < 65)
    df_state['fail_renal'] = (df_state['creatinine'] > 2.0) | (df_state['urine_ml_kg'] < 0.5)
    df_state['fail_resp'] = (df_state['pf_ratio'] < 300) 
    
    # Group by Stay and Date to count "Failure Days"
    df_state['day_index'] = (df_state['chart_hour'] - df_state.groupby('stay_id')['chart_hour'].transform('min')).dt.days
    
    daily_fail = df_state[df_state['day_index'] < 30].groupby(['stay_id', 'day_index'])[['fail_cv', 'fail_renal', 'fail_resp']].max().reset_index()
    failed_counts = daily_fail.groupby('stay_id')[['fail_cv', 'fail_renal', 'fail_resp']].sum().reset_index()
    
    df_final = df_final.merge(failed_counts, on='stay_id', how='left').fillna(0)
    
    died_within_30 = df_final['mortality_30d'] == 1
    
    for organ in ['cv', 'renal', 'resp']:
        col_fail = f'fail_{organ}'
        col_free = f'{organ}_free_days'
        df_final[col_free] = 30 - df_final[col_fail]
        df_final[col_free] = df_final[col_free].clip(lower=0)
        df_final.loc[died_within_30, col_free] = 0
    # Investigation Script: Identifying Dropped IDs before run the plotting loop
    # Check alignment for each phenotype
    for group in df_final['Phenotype'].unique():
        subset = df_final[df_final['Phenotype'] == group]
        valid_n = subset['T_90'].notnull().sum()
        print(f"      Group: {group:20} | Total N: {len(subset):4} | Plottable N: {valid_n:4}")
        
        # Identify specifically who is missing dod data
        missing_info = subset[(subset['hospital_expire_flag'] == 1) & (subset['dod'].isna())]
        if not missing_info.empty:
            print(f"        Warning: {len(missing_info)} deaths missing DOD timestamps.")

    # ==============================================================================
    # 4. KAPLAN-MEIER SURVIVAL ANALYSIS (With Risk Table)
    # ==============================================================================
    print("   -> Generating Kaplan-Meier Curves...")

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 1. Prepare Time (T) and Event (E)
    # Fill NaN (Survivors) with 90 days
    # T = df_final['days_to_death'].fillna(90)
    T = df_final['T_90']  # Use the corrected, clipped duration
    # Clip at 90 days (Censoring)
    T = T.clip(upper=90)
    # FIX: Ensure no negatives passed to fitter
    T = T.clip(lower=0) 
    
    E = df_final['E_90']  # Use the 90-day mortality event flag
    
    groups = sorted(df_final['Phenotype'].dropna().unique())
    kmf_objects = []
    results_table = []
    
    for group in groups:
        kmf = KaplanMeierFitter()
        mask = df_final['Phenotype'] == group
        
        # Fit & Plot
        kmf.fit(T[mask], event_observed=E[mask], label=group)
        kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2.5)
        
        kmf_objects.append(kmf)
        
        # Print Debug info to confirm alignment
        print(f"      [{group}] N_Total: {len(df_final[mask])} | N_Plotted: {len(kmf.event_table)}")

        # Collect stats for summary CSV
        subset = df_final[mask]
        stats = {
            'Phenotype': group,
            'N': len(subset),
            'Mortality_30d (%)': subset['mortality_30d'].mean() * 100,
            'Mortality_90d (%)': subset['mortality_90d'].mean() * 100,
            'MCS_Rate (%)': subset['mcs_flag'].mean() * 100,
            'RRT_Rate (%)': subset['rrt_flag'].mean() * 100,
            'Arrest_Rate (%)': subset['arrest_flag'].mean() * 100,
            'CV_Free_Days': subset['cv_free_days'].mean(),
            'Renal_Free_Days': subset['renal_free_days'].mean()
        }
        results_table.append(stats)

    # Add Risk Table
    add_at_risk_counts(*kmf_objects, ax=ax, rows_to_show=['At risk'])
    
    plt.title('90-Day Survival by Clinical Phenotype', fontsize=16, fontweight='bold')
    plt.xlabel('Days since Admission', fontsize=14)
    plt.ylabel('Survival Probability', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "KM_Survival_Curves_90d_new.png"), dpi=300)
    
    # ==============================================================================
    # 5. SAVE SUMMARY STATISTICS
    # ==============================================================================
    print("   -> Saving Summary Statistics...")
    res_df = pd.DataFrame(results_table).set_index('Phenotype').round(2)
    res_csv = os.path.join(OUTPUT_DIR, "Endpoint_Analysis_Summary.csv")
    res_df.to_csv(res_csv)
    
    print("\n📊 --- ENDPOINT ANALYSIS SUMMARY ---")
    print(res_df.to_string())
    print("-" * 40)
    
    con.close()

if __name__ == "__main__":
    run_endpoint_analysis()


# PS C:\Users\howar\Desktop\DHLAB_code\cardiogenic_shock\0128_NEW> poetry run python .\006_endpoint_analysis_new.py
# 📈 [Step 6] Running Study Endpoints & Subgroup Analysis...
#    -> Loading Cohort & State Space...
#    -> Classifying Phenotypes...
# C:\Users\howar\Desktop\DHLAB_code\cardiogenic_shock\0128_NEW\006_endpoint_analysis_new.py:81: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
#   df_state.groupby('stay_id')['hypo_instant'].shift(1).fillna(False)
#    -> Extracting Adverse Events (RRT, Transplant, Arrest)...
#    -> Calculating Mortality & Corrected Survival Times...
#       [Debug] Patients with invalid times: 0
#    -> Calculating Mortality & Organ-Free Days...
#       Group: Classic Shock        | Total N:  736 | Plottable N:  736
#       Group: Normotensive Cardiogenic Shock | Total N:  177 | Plottable N:  177
#       Group: Neither              | Total N:  298 | Plottable N:  298
#       Group:  Compensated Shock   | Total N: 1160 | Plottable N: 1160
#    -> Generating Kaplan-Meier Curves...
#       [ Compensated Shock ] N_Total: 1160 | N_Plotted: 942
#       [Classic Shock] N_Total: 736 | N_Plotted: 652
#       [Neither] N_Total: 298 | N_Plotted: 245
#       [Normotensive Cardiogenic Shock] N_Total: 177 | N_Plotted: 140
#    -> Saving Summary Statistics...

# 📊 --- ENDPOINT ANALYSIS SUMMARY ---
#                                    N  Mortality_30d (%)  Mortality_90d (%)  MCS_Rate (%)  RRT_Rate (%)  Arrest_Rate (%)  CV_Free_Days  Renal_Free_Days
# Phenotype
#  Compensated Shock              1160              26.72              34.05         20.86          7.67             8.88         18.22            16.65
# Classic Shock                    736              61.14              68.07         19.16         41.58            15.90          7.74             6.63
# Neither                          298              22.82              31.88          0.00          3.02             7.38         22.41            19.97
# Normotensive Cardiogenic Shock   177              36.72              48.02          6.78         18.08             8.47         16.14            13.80
# ----------------------------------------

# action required:
#  update term cryptic shock into normotensive cardiogenic shock and  Preserved Perfusion into compensated shock (V)
# and update thesis overleaf