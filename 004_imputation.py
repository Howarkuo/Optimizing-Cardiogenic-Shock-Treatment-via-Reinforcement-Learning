import duckdb
import os
import pandas as pd
import numpy as np


# table created : 
# rl_state_space, 

# ==================================================
#  Processing Table: 'rl_state_space'
# ==================================================
#  Available Columns: ['sodium', 'pcwp', 'age', 'chart_hour', 'hr', 'mcs_active', 'spo2', 'temp', 'aado2', 'fio2', 'calcium', 'creatinine', 'lactate', 'subject_id', 'urine_output', 'glucose', 'po2', 'magnesium', 'pco2', 'map', 'bsa', 'vaso_rate', 'gender', 'cvp', 'potassium', 'ph', 'ci', 'sbp', 'stay_id', 'resp', 'hadm_id']

# --- Requested Statistics ---
#  distinct_patients  distinct_admissions  distinct_icu_stays
#               1848                 1969                2371


# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mimic_shock.db")

def run_step_4():
    print(f" [Step 4] Building Hourly Grid & Imputing Data...")
    
    if not os.path.exists(DB_PATH):
        print("❌ Error: DB not found. Run previous steps first.")
        return

    con = duckdb.connect(DB_PATH)

# ==============================================================================
    # 1. GENERATE HOURLY SKELETON (1 Row per Hour per Patient)
    # ==============================================================================
    print("   -> [1/4] Generating Hourly Skeleton...")
    con.execute("""
    CREATE OR REPLACE TEMP TABLE skeleton AS
    WITH RECURSIVE recursive_time AS (  -- <--- KEY FIX: Added 'RECURSIVE' here
        SELECT 
            stay_id,
            date_trunc('hour', intime) as chart_hour,
            date_trunc('hour', outtime) as end_hour
        FROM cohort
        
        UNION ALL
        
        SELECT 
            stay_id,
            chart_hour + INTERVAL 1 HOUR,
            end_hour
        FROM recursive_time
        WHERE chart_hour < end_hour
    )
    SELECT DISTINCT stay_id, chart_hour 
    FROM recursive_time
    """)

    # ==============================================================================
    # 2. JOIN & AGGREGATE (Severity Logic: MAX/MIN)
    # ==============================================================================
    print("   -> [2/4] Joining Data with Severity Aggregation...")
    
    # We create a wide table in SQL before loading to Pandas
    query = """
    SELECT 
        s.stay_id,
        s.chart_hour,
        
        -- STATIC FEATURES (Carry forward)
        MAX(c.subject_id) as subject_id,
        MAX(c.hadm_id) as hadm_id,
        MAX(c.anchor_age) as age,
        MAX(c.gender) as gender,
        
        -- VITALS (Severity Logic: Min BP/SpO2, Max HR/RR)
        MIN(v.sbp) as sbp,           -- Hypotension is the risk
        MIN(v.map) as map,           -- Hypotension is the risk
        MIN(v.spo2) as spo2,         -- Hypoxia is the risk
        MAX(v.heart_rate) as hr,     -- Tachycardia
        MAX(v.resp_rate) as resp,    -- Tachypnea
        MAX(v.temperature) as temp,  -- Fever (Infection risk)
        MAX(v.cvp) as cvp,
        MAX(v.pcwp) as pcwp,

        -- LABS (Severity: Max Lactate/Cr, Min pH/pO2)
        MAX(l.lactate) as lactate,
        MAX(l.creatinine) as creatinine, -- Kidney Failure
        MAX(l.glucose) as glucose,
        MAX(l.potassium) as potassium,
        MIN(l.sodium) as sodium,         -- Hyponatremia common in HF
        MIN(l.magnesium) as magnesium,   -- Low Mg causes arrhythmia
        MIN(l.calcium_ionized) as calcium, -- Low Ca causes weak heart
        MIN(l.ph) as ph,                 -- Acidosis
        MIN(l.po2) as po2,               -- Hypoxia
        MAX(l.pco2) as pco2,             -- Hypercapnia
        MAX(l.fio2_lab) as fio2,         -- High O2 requirement
        MAX(l.aado2) as aado2,

        -- BSA & CI
        MAX(b.bsa) as bsa,
        MIN(b.cardiac_index) as ci,      -- Low CI is shock

        -- URINE (Sum)
        MAX(u.urine_ml) as urine_output, 

        -- VASOPRESSORS (Max Rate in that hour)
        MAX(vp.rate_ned) as vaso_rate,

        -- MCS (Binary Status)
        MAX(CASE WHEN m.device_type IS NOT NULL THEN 1 ELSE 0 END) as mcs_active

    FROM skeleton s
    INNER JOIN cohort c ON s.stay_id = c.stay_id
    LEFT JOIN vitals_processed v ON s.stay_id = v.stay_id AND s.chart_hour = v.charttime
    LEFT JOIN labs_processed l ON s.stay_id = l.stay_id AND date_trunc('hour', l.charttime) = s.chart_hour
    LEFT JOIN bsa_ci_processed b ON s.stay_id = b.stay_id -- Static per admission usually
    LEFT JOIN urine_processed u ON s.stay_id = u.stay_id AND s.chart_hour = u.chart_hour
    LEFT JOIN vaso_processed vp ON s.stay_id = vp.stay_id 
         AND s.chart_hour >= date_trunc('hour', vp.starttime) 
         AND s.chart_hour < date_trunc('hour', vp.endtime)
    LEFT JOIN mcs_processed m ON s.stay_id = m.stay_id 
         AND s.chart_hour >= date_trunc('hour', m.starttime) 
         AND s.chart_hour < date_trunc('hour', m.endtime)
         
    GROUP BY s.stay_id, s.chart_hour
    ORDER BY s.stay_id, s.chart_hour
    """
    
    # Load into Pandas for complex imputation
    df = con.execute(query).fetchdf()
    print(f"      Loaded {len(df):,} rows into memory.")

    # ==============================================================================
    # 3. IMPUTATION (Custom Logic)
    #    Rule 1: Gap = 1hr -> LOCF
    #    Rule 2: Gap > 1hr -> Median
    #    Rule 3: Start Gap -> Median
    # ==============================================================================
    print("   -> [3/4] Applying Custom Imputation Rules...")

    # Define dynamic columns (exclude IDs and Time)
    cols_to_impute = [
        'sbp', 'map', 'spo2', 'hr', 'resp', 'temp', 'cvp', 'pcwp',
        'lactate', 'creatinine', 'glucose', 'potassium', 'sodium', 'magnesium', 'calcium',
        'ph', 'po2', 'pco2', 'fio2', 'aado2', 'bsa', 'ci'
    ]
    
    # Pre-calculate Population Medians (Rule 2 & 3 fallback)
    # Note: We use nanmedian to ignore missing values
    pop_medians = df[cols_to_impute].median()
    
    # Fill Urine/Vaso/MCS with 0 first (logic: if null, assume 0 output/rate)
    df['urine_output'] = df['urine_output'].fillna(0)
    df['vaso_rate'] = df['vaso_rate'].fillna(0)
    df['mcs_active'] = df['mcs_active'].fillna(0)

    # Function to apply the "LOCF-1 then Median" logic per patient
    def custom_imputer(group):
        # 1. LOCF with limit=1 (Satisfies: "Gap=1 -> Carry Forward")
        #    If Gap > 1, the second NaN remains NaN.
        group[cols_to_impute] = group[cols_to_impute].ffill(limit=1)
        
        # 2. Fill remaining NaNs (Gaps > 1 and Start Gaps) with Population Median
        group[cols_to_impute] = group[cols_to_impute].fillna(pop_medians)
        
        return group

    # Apply per patient (stay_id)
    # tqdm could be added here if it's slow, but for <2k patients it's fast.
    df = df.groupby('stay_id', group_keys=False).apply(custom_imputer)

    # ==============================================================================
    # 4. SAVE FINAL TABLE
    # ==============================================================================
    print("   -> [4/4] Saving 'rl_state_space' to DuckDB...")
    
    con.execute("DROP TABLE IF EXISTS rl_state_space")
    con.execute("CREATE TABLE rl_state_space AS SELECT * FROM df")
    
    # Final Check
    row_count = con.execute("SELECT COUNT(*) FROM rl_state_space").fetchone()[0]
    col_count = len(df.columns)
    
    print("-" * 40)
    print(f"✅ State Space Built Successfully!")
    print(f"   📦 Table: rl_state_space")
    print(f"   📏 Dimensions: {row_count:,} rows x {col_count} columns")
    print(f"   🧠 Imputation Strategy Applied: LOCF(1h) -> Population Median")
    print("-" * 40)
    
    con.close()

if __name__ == "__main__":
    run_step_4()



# ==================================================
#  Processing Table: 'rl_state_space'
# ==================================================
#  Available Columns: ['gender', 'temp', 'sodium', 'bsa', 'subject_id', 'ci', 'glucose', 'pcwp', 'aado2', 'age', 'vaso_rate', 'creatinine', 'resp', 'ph', 'mcs_active', 'lactate', 'po2', 'stay_id', 'pco2', 'sbp', 'calcium', 'cvp', 'urine_output', 'magnesium', 'spo2', 'chart_hour', 'hr', 'fio2', 'hadm_id', 'potassium', 'map']
#  Enter columns to select for 'rl_state_space' (comma-separated) or Enter for ALL: 

# --- Requested Statistics ---
#  distinct_patients  distinct_admissions  distinct_icu_stays
#               1848                 1969                2371

# --- Data Preview ---
# gender      temp  sodium      bsa  subject_id       ci  glucose  pcwp  aado2  age  vaso_rate  creatinine  resp   ph  mcs_active  lactate   po2  stay_id  pco2   sbp  calcium  cvp  urine_output  magnesium  spo2          chart_hour    hr  fio2  hadm_id  potassium  map
#      M 36.833333   137.0 1.915862    12207593 1.623696 7.388889  20.0   5.73   43        0.0         1.6  20.0 7.39           0      1.8 101.0 30000646  39.0 107.0     1.11 13.0           0.0        2.2  97.0 2194-04-29 01:00:00  88.0  50.0 22795209        4.1 72.0
#      M 36.833333   138.0 1.915862    12207593 1.623696 5.666667  20.0   5.73   43        0.0         0.9  33.0 7.39           0      1.8 101.0 30000646  39.0 111.0     1.11 13.0           0.0        2.2  98.0 2194-04-29 02:00:00 102.0  50.0 22795209        3.5 75.0
#      M 36.833333   138.0 1.915862    12207593 1.623696 5.666667  20.0   5.73   43        0.0         0.9  19.0 7.39           0      1.8 101.0 30000646  39.0  97.0     1.11 13.0           0.0        2.2  94.0 2194-04-29 03:00:00  97.0  50.0 22795209        3.5 67.0
#      M 36.833333   137.0 1.915862    12207593 1.623696 7.388889  20.0   5.73   43        0.0         1.6  18.0 7.39           0      1.8 101.0 30000646  39.0  98.0     1.11 13.0           0.0        2.2  98.0 2194-04-29 04:00:00  93.0  50.0 22795209        4.1 67.0
#      M 37.111111   137.0 1.915862    12207593 1.623696 7.388889  20.0   5.73   43        0.0         1.6  24.0 7.39           0      1.8 101.0 30000646  39.0  98.0     1.11 13.0         700.0        2.2  98.0 2194-04-29 05:00:00  87.0  50.0 22795209        4.1 73.0