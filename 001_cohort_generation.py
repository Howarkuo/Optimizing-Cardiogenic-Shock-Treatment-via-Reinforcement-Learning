# Purpose: Filters the population.
# thesis Methodology -data source and management, study population(inclusion criteria, cohort filtering logic), cohort characteristic
# https://www.overleaf.com/project/69806b86c22616fef1807b03
# Your Logic Applied:

# ICD Codes: 78551, 99801, R570, T8111.

# Exclusions: Age < 18, LOS < 24h.

# Patient count: Multiple ICU stay would be considered approbable and count as different entities.


# connects to raw CSVs and initialize the mimic_shock.db file
# create VIEW for all huge CSv files so you don't load them into RAM

# destination to create .duckdb file and unwrapp csv file-  C:\Users\howar\Desktop\DHLAB_code\cardiogenic_shock\0128_NEW

# Define Inputs: Point to the raw CSVs (diagnoses_icd, patients, admissions, icustays).
# Step A (Shock): Filter diagnoses_icd for your specific codes -> create tmp_shock_adm.
# Step B (Adults): Filter patients for Age >= 18 -> create tmp_adults.
# Step C (Combine): Join icustays (base) + tmp_shock_adm + tmp_adults.
# Step D (Filter): Keep only rows where seq=1 (first stay) and los >= 1.0 (24 hours).


#table created: 
# 
# "tmp_shock_adm", "hadm_id", "Admissions with Shock","tmp_adults", "subject_id", "Adult Patients","tmp_icu_raw", "stay_id", "Total ICU Stays (Raw)"

#logic 
# Did the patient have any ICU stay longer than 24 hours ? 

# function
# setup_datafolder , getpath(), step1selectcohort(), 

# result: 1848 subid, 1969 admid 2371 stayid / 1.2 stays per patientadmin 
# Previous Run:2,531 Stays $\div$ 2,105 Admissions $\approx$ 1.20 stays per patient.

import duckdb
import os
import sys
import zipfile

# ==========================================
#  CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mimic_shock.db")
ZIP_NAME = "mimic-iv-2.2.zip"

def setup_data_folder():
    """
    Ensures the raw data is unzipped and ready to read.
    Returns the path to the unzipped 'mimic-iv-2.2' folder.
    """
    # 1. Search for the unzipped folder
    candidates = [
        os.path.join(BASE_DIR, "mimic-iv-2.2"),
        os.path.join(BASE_DIR, "mimic-iv-2.2", "mimic-iv-2.2") # Handle double nesting
    ]
    
    for path in candidates:
        # Check if critical file exists in this folder to verify it's valid
        check_file = os.path.join(path, "hosp", "patients.csv.gz")
        check_file_csv = os.path.join(path, "hosp", "patients.csv")
        
        if os.path.exists(check_file) or os.path.exists(check_file_csv):
            print(f" Found raw data at: {path}")
            return path

    # 2. If not found, look for the ZIP file and extract it
    zip_path = os.path.join(BASE_DIR, ZIP_NAME)
    if os.path.exists(zip_path):
        print(f" Found '{ZIP_NAME}'. Extraction needed.")
        print("    Extracting files... (This takes 2-5 minutes, please wait)")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(BASE_DIR)
            print("   Extraction complete!")
            
            # Recursive check after extraction
            return setup_data_folder() 
        except Exception as e:
            print(f" Extraction failed: {e}")
            sys.exit(1)
            
    print(" ERROR: Could not find 'mimic-iv-2.2' folder OR 'mimic-iv-2.2.zip'.")
    print(f"   Please ensure the zip file is located at: {BASE_DIR}")
    sys.exit(1)

def get_path(root_path, subfolder, filename):
    """Finds the raw file (csv or gz) in the raw folder."""
    candidates = [
        os.path.join(root_path, subfolder, filename),
        os.path.join(root_path, subfolder, filename + ".gz"),
        os.path.join(root_path, filename),
        os.path.join(root_path, filename + ".gz")
    ]
    for path in candidates:
        if os.path.exists(path):
            return path.replace('\\', '/')
    
    print(f" CRITICAL ERROR: Could not find {filename}")
    print(f"   Checked locations inside: {root_path}")
    sys.exit(1)

def step_1_select_cohort():
    # 1. Ensure Data is Ready
    raw_data_path = setup_data_folder()
    
    print(f"[Step 1] Identifying Cohort...")
    con = duckdb.connect(DB_PATH)
    # helper function for printing counts
    def print_step_count(table_name, id_column, label):
        count = con.execute(f"SELECT COUNT(DISTINCT {id_column}) FROM {table_name}").fetchone()[0]
        print(f"    > [{label}] Unique {id_column}s: {count:,}")

    # -------------------------------------------------------
    # 2. Identify Shock Admissions (ICD Codes)
    # Select distinct hadm_id instead of subject_id since we are treating every admission (visit / hospital stay) as unique entity
    # -------------------------------------------------------
    print("   -> [1/4] Scanning Diagnoses (Shock Codes)...")
    diag_path = get_path(raw_data_path, "hosp", "diagnoses_icd.csv")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tmp_shock_adm AS
        SELECT DISTINCT hadm_id
        FROM read_csv_auto('{diag_path}')
        WHERE icd_code IN ('78551', '99801', 'R570', 'T8111')
    """)
    print_step_count("tmp_shock_adm", "hadm_id", "Admissions with Shock")

    # -------------------------------------------------------
    # 3. Identify Adults + Get Date of Death (dod)
    # -------------------------------------------------------
    print("   -> [2/4] Scanning Patients (Age & Mortality)...")
    pat_path = get_path(raw_data_path, "hosp", "patients.csv")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tmp_adults AS
        SELECT 
            subject_id, 
            anchor_age, 
            gender, 
            anchor_year,
            dod
        FROM read_csv_auto('{pat_path}')
        WHERE anchor_age > 18
    """)
    print_step_count("tmp_adults", "subject_id", "Adult Patients")

    # -------------------------------------------------------
    # 4. Get Hospital Expiry Flag from Admissions
    # -------------------------------------------------------
    print("   -> [3/4] Scanning Admissions (Outcome Labels)...")
    adm_path = get_path(raw_data_path, "hosp", "admissions.csv")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tmp_admissions AS
        SELECT 
            hadm_id, 
            hospital_expire_flag
        FROM read_csv_auto('{adm_path}')
    """)
    print_step_count("tmp_admissions", "hadm_id", "Total Admissions Available")

    # -------------------------------------------------------
    # 5. Create FINAL COHORT Table
    # -------------------------------------------------------
    print("   -> [4/4] Filtering ICU Stays & Joining...")
    icu_path = get_path(raw_data_path, "icu", "icustays.csv")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tmp_icu_raw AS
        SELECT * FROM read_csv_auto('{icu_path}')
    """)
    print_step_count("tmp_icu_raw", "stay_id", "Total ICU Stays (Raw)")

    con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
        WITH ordered_stays AS (
            SELECT 
                i.subject_id, i.hadm_id, i.stay_id, 
                i.intime, i.outtime, i.los,
            FROM read_csv_auto('{icu_path}') i
        )
        SELECT 
            o.subject_id, o.hadm_id, o.stay_id, 
            o.intime, o.outtime, o.los,
            a.anchor_age, a.gender, a.anchor_year,
            a.dod,
            adm.hospital_expire_flag
        FROM ordered_stays o
        INNER JOIN tmp_adults a ON o.subject_id = a.subject_id
        INNER JOIN tmp_shock_adm s ON o.hadm_id = s.hadm_id
        INNER JOIN tmp_admissions adm ON o.hadm_id = adm.hadm_id
        WHERE o.los >= 1.0           
            
    """)

    # -------------------------------------------------------
    # VERIFICATION
    # -------------------------------------------------------
    # Count distinct admissions
    count = con.execute("SELECT COUNT(*) FROM cohort").fetchone()[0]
    count_adm = con.execute("SELECT COUNT(DISTINCT hadm_id) FROM cohort").fetchone()[0]

    # Count distinct patients
    count_sub = con.execute("SELECT COUNT(DISTINCT subject_id) FROM cohort").fetchone()[0]
    count_stay = con.execute("SELECT COUNT(DISTINCT stay_id) FROM cohort").fetchone()[0]


    print(f"Cohort Created Successfully!")
    print(f"    Total Rows (ICU Stays): {con.execute('SELECT COUNT(*) FROM cohort').fetchone()[0]:,}")
    print(f"    Unique Admissions:      {count_adm:,}")  # This will be <= 2,296
    print(f"    Unique Patients:        {count_sub:,}")
    print(f"    Unique ICU stay:        {count_stay:,}")

    
    print("-" * 40)
    if count > 0:
        mortality = con.execute("SELECT AVG(hospital_expire_flag) FROM cohort").fetchone()[0]
        print(f"Cohort Created Successfully!")
        print(f"    Total Patients: {count}")
        print(f"   Mortality Rate: {mortality:.1%}")
    else:
        print(" Warning: Cohort is 0. Check your ICD codes.")
        
    print(f"    Saved to: {DB_PATH}")
    print("-" * 40)
    
    con.close()

if __name__ == "__main__":
    step_1_select_cohort()


# python .\01_cohort_generation.pyode\cardiogenic_shock\0128_NEW>
#  Found raw data at: C:\Users\howar\Desktop\DHLAB_code\cardiogenic_shock\0128_NEW\mimic-iv-2.2
# [Step 1] Identifying Cohort...
#    -> [1/4] Scanning Diagnoses (Shock Codes)...
#     > [Admissions with Shock] Unique hadm_ids: 2,296
#    -> [2/4] Scanning Patients (Age & Mortality)...
#     > [Adult Patients] Unique subject_ids: 295,780
#    -> [3/4] Scanning Admissions (Outcome Labels)...
#     > [Total Admissions Available] Unique hadm_ids: 431,231
#    -> [4/4] Filtering ICU Stays & Joining...
#     > [Total ICU Stays (Raw)] Unique stay_ids: 73,181
# Cohort Created Successfully!
#     Total Rows (ICU Stays): 2,371
#     Unique Admissions:      1,969
#     Unique Patients:        1,848
#     Unique ICU stay:        2,371
# ----------------------------------------
# Cohort Created Successfully!
#     Total Patients: 2371
#    Mortality Rate: 32.7%
#     Saved to: C:\Users\howar\Desktop\DHLAB_code\cardiogenic_shock\0128_NEW\mimic_shock.db
