# 02_extract_raw_data
# read massive raw CSVs, filter them against cohort table (1848, 1969, 2371) and save the filtered tables in database

# input: raw CSVs in mimic-iv-2.2/
# output: A larger mimic_shock.db file containing 7 new filtered tables.

# functionsL get_path, step2_extract_data(), 

import duckdb
import os
import time

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check for nested folder structure which sometimes happens with unzip
CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, "mimic-iv-2.2"),
    os.path.join(BASE_DIR, "mimic-iv-2.2", "mimic-iv-2.2")
]
RAW_DATA_PATH = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
DB_PATH = os.path.join(BASE_DIR, "mimic_shock.db")

def get_path(subfolder, filename):
    """Finds the raw file (csv or gz) in the raw folder."""
    if not RAW_DATA_PATH:
        raise FileNotFoundError("Could not find mimic-iv-2.2 folder")
        
    candidates = [
        os.path.join(RAW_DATA_PATH, subfolder, filename),
        os.path.join(RAW_DATA_PATH, subfolder, filename + ".gz"),
        os.path.join(RAW_DATA_PATH, filename),
        os.path.join(RAW_DATA_PATH, filename + ".gz")
    ]
    for path in candidates:
        if os.path.exists(path):
            return path.replace('\\', '/')
    raise FileNotFoundError(f" Missing file: {filename}")

def step_2_extract_data():
    print(f" [Step 2] Extracting Clinical Data for Cohort...")
    
    if not os.path.exists(DB_PATH):
        print(" Error: DB not found. Run '01_cohort_generation.py' first.")
        return

    con = duckdb.connect(DB_PATH)

    # Check if cohort exists
    try:
        count = con.execute("SELECT COUNT(*) FROM cohort").fetchone()[0]
        print(f"   -> Targeted Cohort: {count} patients.")
    except:
        print(" Error: 'cohort' table missing. Run Step 1 first.")
        return

    # List of tables to process. 
    # Format: (Table Name, Subfolder, Filename, Join Key)
    tables_to_extract = [
        ("admissions",      "hosp", "admissions.csv",      "hadm_id"),
        ("labevents",       "hosp", "labevents.csv",       "hadm_id"),
        ("icustays",        "icu",  "icustays.csv",        "stay_id"),
        ("procedureevents", "icu",  "procedureevents.csv", "stay_id"),
        ("inputevents",     "icu",  "inputevents.csv",     "stay_id"),
        ("outputevents",    "icu",  "outputevents.csv",    "stay_id"),
        ("chartevents",     "icu",  "chartevents.csv",     "stay_id") # The Big One
    ]

    for tbl, sub, fname, key in tables_to_extract:
        print(f"   ... Processing {tbl} (Filtering by {key})...")
        
        try:
            fpath = get_path(sub, fname)
            
            # Dynamic Join Condition
            if key == "hadm_id":
                join_cond = "t.hadm_id = c.hadm_id"
            elif key == "stay_id":
                join_cond = "t.stay_id = c.stay_id"
            
            # The Extraction Query
            query = f"""
                CREATE OR REPLACE TABLE {tbl} AS
                SELECT t.* FROM read_csv_auto('{fpath}') t
                INNER JOIN cohort c ON {join_cond}
            """
            
            start_time = time.time()
            con.execute(query)
            elapsed = time.time() - start_time
            
            # Verify
            rows = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"        Saved {rows:,} rows (Time: {elapsed:.1f}s)")
            
        except Exception as e:
            print(f"       Failed to extract {tbl}: {e}")

    print("\n Data Extraction Complete!")
    print(f"   Database now contains all raw data for your cohort.")
    con.close()

if __name__ == "__main__":
    step_2_extract_data()