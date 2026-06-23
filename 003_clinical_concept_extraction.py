# Data Manipulation Phase (DML)

# Purpose: Extract HR, BP, SpO2, RespRate, Temp.
# Logic Applied
# BP Priority: COALESCE(invasive_bp, non_invasive_bp).SpO2: Range $0 < SpO2 \le 100$.Outliers: Filter HR > 300, SBP > 400.
#  Data Range:
#  1. ICU/Chartevent/valuenum
#  HR : 0~300, SpO2 0~100, Resp Rate 0~70 , Temp F[70-120] -> C, C[10-50], BP Invasive 0 < Sys < 400, 0 < Dia < 300, 0 < Mean < 300),BP Non-Invasive (Same limits), #    Logic: Weight > 0, Height 120-230cm, Dubois Formula, Cardiac Output range 0.5-20 
#  Reasoning: Prevent typo from the nurses
# 2. hosp/labevents/valuenum
# glucose <= 10000, lactate <= 30 , FiO2 20~100, 
# No range applied: ph, Creatinine, po2, pco2, 
# Reasoning: Prevent contaminated samples
# 3. ICU/outputevents/value
# urine output: Foley catheter, voided urine, condom catheter, ileoconduit, suprapubic catheter, right nephrostomy, left nephrostomy, straight catheter, right ureteral stent, left ureteral stent , (genitourinary irrigant urine volume out - genitourinary irrigant urine volume in)
# 4. Derived Parameters
# Debois BSA: 0.07 * weight^ 0.425 * Height ^ 0.725 (Assume no infants and obese adults, if obese will underestimate)
#  Functions
# run_extraction()

# tables
# vitals_processed, labs_processed, 

import duckdb
import os

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mimic_shock.db")

def run_extraction():
    print(f" [Step 3] Extracting Clinical Concepts with Strict Logic...")
    
    if not os.path.exists(DB_PATH):
        print(" Error: DB not found. Run Step 1 & 2 first.")
        return

    con = duckdb.connect(DB_PATH)

    # ==============================================================================
    # 1. VITALS (+ CVP, PCWP for SVR calculation)
    # ==============================================================================
    print("   -> [1/6] Processing Vitals (HR, BP, SpO2, CVP, PCWP)...")
    con.execute("""
    CREATE OR REPLACE TABLE vitals_processed AS
    WITH raw_vitals AS (
        SELECT 
            ce.stay_id, 
            ce.charttime,
            ce.itemid,
            ce.valuenum
        FROM chartevents ce
        INNER JOIN cohort c ON ce.stay_id = c.stay_id -- Ensure ID consistency
        WHERE ce.itemid IN (
            220045, -- HR
            220277, -- SpO2
            220210, -- Resp Rate
            223761, 223762, -- Temp
            220050, 225309, 220051, 225310, 220052, 225312, -- Invasive BP
            220179, 220180, 220181, -- Non-Invasive BP
            220074, -- CVP (Central Venous Pressure)
            223771  -- PCWP (Pulmonary Capillary Wedge Pressure)
        )
    )
    SELECT 
        c.subject_id, c.hadm_id, rv.stay_id, 
        rv.charttime,
        
        -- Heart Rate
        AVG(CASE WHEN itemid = 220045 AND valuenum BETWEEN 0 AND 300 THEN valuenum END) AS heart_rate,
        -- SpO2
        AVG(CASE WHEN itemid = 220277 AND valuenum BETWEEN 0 AND 100 THEN valuenum END) AS spo2,
        -- Resp Rate
        AVG(CASE WHEN itemid = 220210 AND valuenum BETWEEN 0 AND 70 THEN valuenum END) AS resp_rate,
        -- Temp (F->C)
        AVG(CASE 
            WHEN itemid = 223761 AND valuenum BETWEEN 70 AND 120 THEN (valuenum - 32) * 5/9 
            WHEN itemid = 223762 AND valuenum BETWEEN 10 AND 50 THEN valuenum
        END) AS temperature,

        -- BP Priority: Invasive > Non-Invasive
        COALESCE(
            AVG(CASE WHEN itemid IN (220050, 225309) AND valuenum BETWEEN 0 AND 400 THEN valuenum END), -- SBP Inv
            AVG(CASE WHEN itemid IN (220179) AND valuenum BETWEEN 0 AND 400 THEN valuenum END)          -- SBP NIBP
        ) AS sbp,
        
        COALESCE(
            AVG(CASE WHEN itemid IN (220051, 225310) AND valuenum BETWEEN 0 AND 300 THEN valuenum END), -- DBP Inv
            AVG(CASE WHEN itemid IN (220180) AND valuenum BETWEEN 0 AND 300 THEN valuenum END)          -- DBP NIBP
        ) AS dbp,
        
        COALESCE(
            AVG(CASE WHEN itemid IN (220052, 225312) AND valuenum BETWEEN 0 AND 300 THEN valuenum END), -- MAP Inv
            AVG(CASE WHEN itemid IN (220181) AND valuenum BETWEEN 0 AND 300 THEN valuenum END)          -- MAP NIBP
        ) AS map,

        -- Hemodynamics (Needed for SVR)
        AVG(CASE WHEN itemid = 220074 AND valuenum BETWEEN -10 AND 100 THEN valuenum END) AS cvp,
        AVG(CASE WHEN itemid = 223771 AND valuenum BETWEEN 0 AND 100 THEN valuenum END) AS pcwp

    FROM raw_vitals rv
    INNER JOIN cohort c ON rv.stay_id = c.stay_id
    GROUP BY c.subject_id, c.hadm_id, rv.stay_id, rv.charttime
    """)

  # ==============================================================================
    # 2. LABS (Corrected: Hierarchy for Creatinine)
    #    Priority: Serum (50912) > Whole Blood (52024) > Point-of-Care (229761)
    # ==============================================================================
    print("   -> [2/6] Processing Labs (Creatinine Hierarchy & Electrolytes)...")
    
    con.execute("""
    CREATE OR REPLACE TABLE labs_processed AS
    WITH combined_data AS (
        -- 1. Get Lab Events (Serum & Whole Blood)
        SELECT 
            c.stay_id, c.subject_id, c.hadm_id,
            l.charttime, l.itemid, l.valuenum
        FROM labevents l
        INNER JOIN cohort c ON l.hadm_id = c.hadm_id
        WHERE l.itemid IN (
            50931, 50912, 50971, 50983, 50902, 50960, -- Serum Chem
            50809, 50822, 50824, 50806, 50808,        -- Blood Gas lytes
            50813, 50820, 50821, 50818, 50816, 50802, -- Gases
            52024 -- Creatinine Whole Blood (Lab)
        )
        
        UNION ALL
        
        -- 2. Get Point-of-Care Creatinine from Chart (229761)
        SELECT 
            c.stay_id, c.subject_id, c.hadm_id,
            ce.charttime, ce.itemid, ce.valuenum
        FROM chartevents ce
        INNER JOIN cohort c ON ce.stay_id = c.stay_id
        WHERE ce.itemid = 229761 AND ce.valuenum < 20
    )
    SELECT 
        subject_id, hadm_id, stay_id, 
        charttime,
        
        -- CREATININE HIERARCHY: Serum > Whole Blood > Point-of-Care
        COALESCE(
            AVG(CASE WHEN itemid = 50912 THEN valuenum END), -- Priority 1
            AVG(CASE WHEN itemid = 52024 THEN valuenum END), -- Priority 2
            AVG(CASE WHEN itemid = 229761 THEN valuenum END) -- Priority 3
        ) AS creatinine,

        -- Standard Labs
        AVG(CASE WHEN itemid IN (50931, 50809) AND valuenum <= 2000 THEN valuenum / 18.0 END) AS glucose,
        AVG(CASE WHEN itemid = 50813 AND valuenum <= 30 THEN valuenum END) AS lactate,
        AVG(CASE WHEN itemid = 50820 THEN valuenum END) AS ph,
        
        -- Electrolytes
        AVG(CASE WHEN itemid IN (50971, 50822) AND valuenum BETWEEN 1 AND 10 THEN valuenum END) AS potassium,
        AVG(CASE WHEN itemid IN (50983, 50824) AND valuenum BETWEEN 100 AND 200 THEN valuenum END) AS sodium,
        AVG(CASE WHEN itemid IN (50960) THEN valuenum END) AS magnesium,
        AVG(CASE WHEN itemid = 50808 AND valuenum BETWEEN 0 AND 5 THEN valuenum END) AS calcium_ionized,
        
        -- Gases
        AVG(CASE WHEN itemid = 50821 THEN valuenum END) AS po2,
        AVG(CASE WHEN itemid = 50818 THEN valuenum END) AS pco2,
        MAX(CASE WHEN itemid = 50816 AND valuenum BETWEEN 20 AND 100 THEN valuenum END) AS fio2_lab

    FROM combined_data
    GROUP BY subject_id, hadm_id, stay_id, charttime
    """)
    
    # Calculate AaDO2 (Same as before)
    print("      ... Calculating AaDO2 ...")
    con.execute("""
    ALTER TABLE labs_processed ADD COLUMN aado2 DOUBLE;
    UPDATE labs_processed 
    SET aado2 = (
        (COALESCE(fio2_lab, 21) / 100.0) * (760 - 47) - (pco2 / 0.8)
    ) - po2
    WHERE po2 IS NOT NULL AND pco2 IS NOT NULL;
    """)
 

    # ==============================================================================
    # 3. BSA & CARDIAC INDEX
    # ==============================================================================
    print("   -> [3/6] Calculating BSA & Cardiac Index...")
    con.execute("""
    CREATE OR REPLACE TABLE bsa_ci_processed AS
    WITH ht_wt AS (
        SELECT 
            ce.stay_id,
            COALESCE(
                MAX(CASE WHEN itemid = 226512 AND valuenum > 0 THEN valuenum END),
                AVG(CASE WHEN itemid = 224639 AND valuenum > 0 THEN valuenum END)
            ) as weight_kg,
            AVG(CASE WHEN itemid = 226730 AND valuenum BETWEEN 120 AND 230 THEN valuenum END) as height_cm
        FROM chartevents ce
        INNER JOIN cohort c ON ce.stay_id = c.stay_id
        WHERE itemid IN (226512, 224639, 226730)
        GROUP BY ce.stay_id
    ),
    cardiac_output AS (
        SELECT ce.stay_id, ce.charttime, ce.valuenum as co
        FROM chartevents ce
        INNER JOIN cohort c ON ce.stay_id = c.stay_id
        WHERE itemid IN (220088, 224842, 227543, 228178, 228369) AND valuenum BETWEEN 0.5 AND 20
    )
    SELECT 
        c.subject_id, c.hadm_id, co.stay_id, 
        co.charttime,
        co.co as cardiac_output, -- Kept raw CO for SVR calc later
        (0.007184 * POWER(hw.weight_kg, 0.425) * POWER(hw.height_cm, 0.725)) AS bsa,
        co.co / (0.007184 * POWER(hw.weight_kg, 0.425) * POWER(hw.height_cm, 0.725)) AS cardiac_index,
        hw.weight_kg,
        hw.height_cm,
    FROM cardiac_output co
    INNER JOIN ht_wt hw ON co.stay_id = hw.stay_id
    INNER JOIN cohort c ON co.stay_id = c.stay_id
    WHERE hw.weight_kg > 0 AND hw.height_cm > 0
    """)

    # ==============================================================================
    # 4. URINE OUTPUT
    # ==============================================================================
    print("   -> [4/6] Processing Urine Output...")
    con.execute("""
    CREATE OR REPLACE TABLE urine_processed AS
    SELECT 
        c.subject_id, c.hadm_id, oe.stay_id,
        date_trunc('hour', oe.charttime) as chart_hour,
        SUM(
            CASE 
                WHEN itemid IN (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227489) THEN value
                WHEN itemid = 227488 THEN -1 * value
                ELSE 0
            END
        ) as urine_ml
    FROM outputevents oe
    INNER JOIN cohort c ON oe.stay_id = c.stay_id
    WHERE itemid IN (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489)
    GROUP BY c.subject_id, c.hadm_id, oe.stay_id, chart_hour
    HAVING urine_ml >= 0
    """)

    # ==============================================================================
    # 5. VASOPRESSORS & 6. MCS (Binary Flags)
    # ==============================================================================
    print("   -> [5/6] Processing Vasopressors & MCS...")
    # Vaso
    con.execute("""
    CREATE OR REPLACE TABLE vaso_processed AS
    SELECT c.subject_id, c.hadm_id, ie.stay_id, starttime, endtime, itemid,
        CASE 
            WHEN itemid = 221906 THEN rate           -- Norepi
            WHEN itemid = 221289 THEN rate           -- Epi
            WHEN itemid = 221749 THEN rate / 10.0    -- Phenylephrine
            WHEN itemid = 221662 THEN rate / 100.0   -- Dopamine
            WHEN itemid = 222315 THEN (rate * 2.5)/60.0 -- Vaso for 6410 / 6413 are in units/hour, others unit/min
            ELSE 0 
        END AS rate_ned
    FROM inputevents ie
    INNER JOIN cohort c ON ie.stay_id = c.stay_id
    WHERE itemid IN (221906, 221289, 221749, 221662, 222315) AND rate > 0
    """)

    # MCS
    con.execute("""
    CREATE OR REPLACE TABLE mcs_processed AS
    SELECT c.subject_id, c.hadm_id, pe.stay_id, starttime, endtime,
        CASE 
            WHEN itemid = 224272 THEN 'IABP'
            WHEN itemid IN (224314, 229897) THEN 'Impella'
            WHEN itemid = 224660 THEN 'ECMO'
        END AS device_type
    FROM procedureevents pe
    INNER JOIN cohort c ON pe.stay_id = c.stay_id
    WHERE itemid IN (224272, 224314, 229897, 224660)
    """)

    # ==============================================================================
    # FINAL COUNT
    # ==============================================================================
    print("\n✅ Extraction Complete!")
    tables = ["vitals_processed", "labs_processed", "bsa_ci_processed", "urine_processed", "vaso_processed", "mcs_processed"]
    for t in tables:
        try:
            count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"   📊 {t}: {count:,} rows")
        except:
            print(f"   ⚠️ {t} not created.")
            
    con.close()

if __name__ == "__main__":
    run_extraction()


# PS C:\Users\howar\Desktop\DHLAB_code\cardiogenic_shock\0128_NEW\Helperfunction> poetry run python .\Simple_Clinicalhelper.py          
#  Connected to: C:\Users\howar\Desktop\DHLAB_code\cardiogenic_shock\0128_NEW\mimic_shock.db
# ==================================================
#  Processing Table: 'vitals_processed'
# ==================================================
#  Available Columns: ['resp_rate', 'charttime', 'map', 'temperature', 'pcwp', 'cvp', 'stay_id', 'hadm_id', 'subject_id', 'heart_rate', 'sbp', 'dbp', 'spo2']
#  Enter columns to select for 'vitals_processed' (comma-separated) or Enter for ALL: 

# --- Requested Statistics ---
#  distinct_patients  distinct_admissions  distinct_icu_stays
#               1848                 1969                2371
# ==================================================
#  Processing Table: 'labs_processed'
# ==================================================
#  Available Columns: ['ph', 'magnesium', 'aado2', 'lactate', 'potassium', 'subject_id', 'sodium', 'stay_id', 'glucose', 'calcium_ionized', 'pco2', 'po2', 'creatinine', 'hadm_id', 'charttime', 'fio2_lab']
#  Enter columns to select for 'labs_processed' (comma-separated) or Enter for ALL:

# --- Requested Statistics ---
#  distinct_patients  distinct_admissions  distinct_icu_stays
#               1845                 1966                2368

# --- Data Preview ---
#   ph  magnesium   aado2  lactate  potassium  subject_id  sodium  stay_id   glucose  calcium_ionized  pco2   po2  creatinine  hadm_id           charttime  fio2_lab
# 7.33        NaN -160.52      7.3        3.8    10094679     NaN 36938927 11.444444         
#     0.96  29.0 274.0         NaN 28902523 2138-03-28 16:34:00       NaN
#  NaN        2.2     NaN      NaN        4.3    10094679   146.0 36938927  6.111111         
#      NaN   NaN   NaN         0.5 28902523 2138-03-29 09:53:00       NaN
#  NaN        1.8     NaN      NaN        4.3    10094679   142.0 36938927  7.000000         
#      NaN   NaN   NaN         0.7 28902523 2138-03-30 07:39:00       NaN
# 7.46        NaN     NaN      NaN        NaN    10094679     NaN 36938927       NaN         
#     1.05   NaN   NaN         NaN 28902523 2138-04-02 00:36:00       NaN
#  NaN        2.0     NaN      NaN        3.4    10094679   150.0 36938927  5.833333         
#      NaN   NaN   NaN         1.1 28902523 2138-04-03 17:12:00       NaN

# ==================================================
#  Processing Table: 'bsa_ci_processed'
# ==================================================
#  Available Columns: ['cardiac_output', 'bsa', 'subject_id', 'stay_id', 'cardiac_index', 'hadm_id', 'charttime']
#  Enter columns to select for 'bsa_ci_processed' (comma-separated) or Enter for ALL:        

# --- Requested Statistics ---
#  distinct_patients  distinct_admissions  distinct_icu_stays
#                506                  510                 515

# --- Data Preview ---
#  cardiac_output      bsa  subject_id  stay_id  cardiac_index  hadm_id           charttime  
#             6.8 2.363533    11558942 31255985       2.877049 24509205 2181-08-21 19:52:00  
#             6.2 2.363533    11558942 31255985       2.623192 24509205 2181-08-21 21:00:00  
#             7.9 2.363533    11558942 31255985       3.342454 24509205 2181-08-21 21:24:00  
#             6.4 2.363533    11558942 31255985       2.707811 24509205 2181-08-21 22:00:00  
#             7.0 2.363533    11558942 31255985       2.961668 24509205 2181-08-21 23:00:00  

# ==================================================
#  Processing Table: 'urine_processed'
# ==================================================
#  Available Columns: ['urine_ml', 'chart_hour', 'stay_id', 'hadm_id', 'subject_id']
#  Enter columns to select for 'urine_processed' (comma-separated) or Enter for ALL:

# --- Requested Statistics ---
#  distinct_patients  distinct_admissions  distinct_icu_stays
#               1800                 1916                2294

# --- Data Preview ---
#  urine_ml          chart_hour  stay_id  hadm_id  subject_id
#     350.0 2124-02-28 05:00:00 35337353 20995778    13224283
#     300.0 2124-03-03 02:00:00 35337353 20995778    13224283
#      35.0 2124-02-15 18:00:00 39375310 20995778    13224283
#      30.0 2124-02-14 22:00:00 39375310 20995778    13224283
#     300.0 2124-02-16 04:00:00 39375310 20995778    13224283

# ==================================================
#  Processing Table: 'vaso_processed'
# ==================================================
#  Available Columns: ['endtime', 'stay_id', 'starttime', 'itemid', 'rate_ned', 'hadm_id', 'subject_id']
#  Enter columns to select for 'vaso_processed' (comma-separated) or Enter for ALL:

# --- Requested Statistics ---
#  distinct_patients  distinct_admissions  distinct_icu_stays
#               1569                 1633                1796

# --- Data Preview ---
#             endtime  stay_id           starttime  itemid  rate_ned  hadm_id  subject_id
# 2122-12-12 00:46:00 36498498 2122-12-11 23:41:00  221906  0.409724 28292401    15533907    
# 2122-12-12 01:14:00 36498498 2122-12-12 00:46:00  221906  0.450885 28292401    15533907    
# 2122-12-12 06:48:00 36498498 2122-12-12 01:14:00  221906  0.521198 28292401    15533907    
# 2122-12-06 16:56:00 36498498 2122-12-06 09:02:00  221906  0.181129 28292401    15533907    
# 2122-12-07 06:06:00 36498498 2122-12-06 13:26:00  222315  0.100000 28292401    15533907    

# ==================================================
#  Processing Table: 'mcs_processed'
# ==================================================
#  Available Columns: ['endtime', 'stay_id', 'starttime', 'device_type', 'hadm_id', 'subject_id']
#  Enter columns to select for 'mcs_processed' (comma-separated) or Enter for ALL:

# --- Requested Statistics ---
#  distinct_patients  distinct_admissions  distinct_icu_stays
#                368                  370                 396

# --- Data Preview ---
#             endtime  stay_id           starttime device_type  hadm_id  subject_id
# 2200-01-16 13:17:00 33287216 2200-01-14 13:49:00        IABP 22713043    11304959
# 2200-01-12 17:04:00 37142717 2200-01-05 16:41:00        IABP 22713043    11304959
# 2118-03-10 09:45:00 35647824 2118-03-05 04:00:00        IABP 23112869    11374486
# 2142-12-14 12:50:00 35541349 2142-12-13 08:24:00        IABP 25687779    11458288
# 2134-06-13 10:30:00 35623139 2134-06-11 14:00:00        IABP 24501476    11501869