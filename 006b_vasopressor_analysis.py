import duckdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mimic_shock.db")
PHENO_DIR = os.path.join(BASE_DIR, "phenotype_results")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_vaso_analysis():
    print(f" [Step 6b] Running Vasopressor Dosage Analysis...")
    
    if not os.path.exists(DB_PATH):
        print(" Error: DB not found.")
        return

    con = duckdb.connect(DB_PATH)

    # ==============================================================================
    # 1. LOAD PHENOTYPES FROM CSVs
    # ==============================================================================
    print("   -> Loading Phenotype Lists...")
    pheno_map = []
    
    # # Map filenames to readable labels
    # files = {
    #     "Both_ids.csv": "Classic Shock",
    #     "Hypo_ids.csv": "Preserved Perfusion",
    #     "Perf_ids.csv": "Cryptic Shock",
    #     "Neither_ids.csv": "Neither"
    # }
    files = {
        "Classic_ids.csv": "Classic Shock",
        "Compensated_ids.csv": "Compensated Shock",
        "Normotensive_ids.csv": "Normotensive Cardiogenic Shock",
        "Neither_ids.csv": "Neither"
    }
    
    for fname, label in files.items():
        fpath = os.path.join(PHENO_DIR, fname)
        if os.path.exists(fpath):
            ids = pd.read_csv(fpath)
            ids['Phenotype'] = label
            pheno_map.append(ids)
        else:
            print(f"       Warning: {fname} not found.")
            
    if not pheno_map:
        print(" No phenotype files found. Run Step 5 first.")
        return
        
    df_pheno = pd.concat(pheno_map)
    print(f"      Loaded {len(df_pheno)} patients with phenotypes.")

    # ==============================================================================
    # 2. CALCULATE MAX NED & MORTALITY
    # ==============================================================================
    print("   -> Calculating Max Vasopressor Dose (NED)...")
    
    # We get Mortality from Cohort and Max Rate from State Space
    df_data = con.execute("""
        SELECT 
            r.stay_id,
            MAX(r.vaso_rate) as max_ned,
            MAX(c.hospital_expire_flag) as mortality
        FROM rl_state_space r
        JOIN cohort c ON r.stay_id = c.stay_id
        GROUP BY r.stay_id
    """).fetchdf()
    
    # Merge
    df_final = df_pheno.merge(df_data, on='stay_id', how='inner')
    
    # Label Mortality
    df_final['Survival'] = df_final['mortality'].map({0: 'Survivor', 1: 'Non-Survivor'})

    # ==============================================================================
    # 3. STATISTICAL ANALYSIS (Mann-Whitney U)
    # ==============================================================================
    print("   -> Performing Statistical Tests...")
    
    stats_results = []
    # order = ["Classic Shock", "Cryptic Shock", "Preserved Perfusion", "Neither"]
    # Update to new terminology
    order = ["Classic Shock", "Compensated Shock", "Normotensive Cardiogenic Shock", "Neither"]
    for group in order:
        subset = df_final[df_final['Phenotype'] == group]
        surv = subset[subset['mortality'] == 0]['max_ned']
        non_surv = subset[subset['mortality'] == 1]['max_ned']
        
        # Mann-Whitney U Test (Non-parametric t-test equivalent)
        if len(surv) > 0 and len(non_surv) > 0:
            stat, p_val = mannwhitneyu(surv, non_surv, alternative='two-sided')
        else:
            p_val = np.nan
            
        stats_results.append({
            'Phenotype': group,
            'Survivor_Median_NED': surv.median(),
            'NonSurvivor_Median_NED': non_surv.median(),
            'P_Value': p_val
        })

    df_stats = pd.DataFrame(stats_results)
    print(df_stats.to_string())
    
    # Save Stats
    df_stats.to_csv(os.path.join(OUTPUT_DIR, "vaso_stats_summary.csv"), index=False)

    # ==============================================================================
    # 4. VISUALIZATION
    # ==============================================================================
    print("   -> Generating Boxplot...")
    
    plt.figure(figsize=(12, 7))
    
    # Create Boxplot
    ax = sns.boxplot(x='Phenotype', y='max_ned', hue='Survival', data=df_final, 
                     order=order, palette={'Survivor': '#2ca02c', 'Non-Survivor': '#d62728'},
                     showfliers=False) # Hide extreme outliers for cleaner view
    
    plt.title('Max Vasopressor Dose (NED) by Phenotype & Survival', fontsize=16, fontweight='bold')
    plt.ylabel('Max Norepinephrine Equivalent Dose (mcg/kg/min)', fontsize=12)
    plt.xlabel('')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add P-Value Annotations
    y_max = df_final.groupby('Phenotype')['max_ned'].quantile(0.95).max() # Get a reasonable height for text
    
    for i, row in df_stats.iterrows():
        p = row['P_Value']
        # Format P-value text
        if p < 0.001: txt = "p < 0.001 ***"
        elif p < 0.01: txt = f"p = {p:.3f} **"
        elif p < 0.05: txt = f"p = {p:.3f} *"
        else: txt = f"p = {p:.3f} (ns)"
        
        # Position text above the boxplot group
        plt.text(i, y_max, txt, ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "Vasopressor_Usage_Comparison.png")
    plt.savefig(plot_path, dpi=300)
    
    # Also show the plot window
    plt.show()
    print(f"      Plot saved to: {plot_path}")
    
    con.close()

if __name__ == "__main__":
    run_vaso_analysis()

# PS C:\Users\howar\Desktop\DHLAB_code\cardiogenic_shock\0128_NEW> poetry run python .\006b_vasopressor_analysis.py 
#  [Step 6b] Running Vasopressor Dosage Analysis...
#    -> Loading Phenotype Lists...
#       Loaded 2371 patients with phenotypes.
#    -> Calculating Max Vasopressor Dose (NED)...
#    -> Performing Statistical Tests...
#                         Phenotype  Survivor_Median_NED  NonSurvivor_Median_NED       P_Value
# 0                   Classic Shock             0.300439                0.500626  8.371668e-19
# 1               Compensated Shock             0.100221                0.249898  2.993138e-14
# 2  Normotensive Cardiogenic Shock             0.000000                0.100012  5.822825e-03
# 3                         Neither             0.000000                0.000000  1.000000e+00
#    -> Generating Boxplot...
#       Plot saved to: C:\Users\howar\Desktop\DHLAB_code\cardiogenic_shock\0128_NEW\analysis_results\Vasopressor_Usage_Comparison.png