import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define fitting function: y = y0 + A1 * exp(-x / t1)
def equation1(x, y0, A1, t1):
    return y0 + A1 * np.exp(-x / t1)

# R² calculation
def calculate_r2(y_data, y_fit):
    residuals = y_data - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    return 1 - (ss_res / ss_tot)

# === Paths ===
fitting_folder = r"/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/241211/Remove noise csv/Output/Fitted results"
eg_file_path = r"/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/241211/Remove noise csv/Output/UV-Vis-Eg and Eu.csv"

# === Load Eg info ===
eg_df = pd.read_csv(eg_file_path)
eg_df.columns = eg_df.columns.str.strip()
eg_map = dict(zip(eg_df['Sample Nr'].astype(str), eg_df['UV-Vis calculated Eg (eV)']))

# === Use updated Equation 1 ===
param_list = ["y0", "A1", "t1"]
initial_guess = [0.0, 0.06, 100]
equation_name = "Equation_1_with_y0"
fit_func = equation1

# Output folder
output_folder = os.path.join(fitting_folder, f"Fitting_Results_{equation_name}")
os.makedirs(output_folder, exist_ok=True)

results_df = pd.DataFrame(columns=["File", "Eg", "R_2"] + param_list)

# === Loop through each CSV file ===
csv_files = [f for f in os.listdir(fitting_folder) if f.endswith(".csv")]
for file in csv_files:
    file_path = os.path.join(fitting_folder, file)
    print(f"Processing {file}...")

    sample_number = ''.join(filter(str.isdigit, file))
    Eg = eg_map.get(sample_number, np.nan)
    if np.isnan(Eg):
        print(f"  Skipped: Eg not found for sample {sample_number}")
        continue

    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    x = data['Time']
    y = data['Peak 2 Energy(eV)']

    # ✅ 正确：计算 Peak Shift 为每一行减去第一行
    y_shift = y - y.iloc[0]

    try:
        params, _ = curve_fit(fit_func, x, y_shift, p0=initial_guess, maxfev=10000)
        y_fit = fit_func(x, *params)
        r2 = calculate_r2(y_shift, y_fit)

        # Save fit results
        row = {"File": file, "Eg": Eg, "R_2": r2}
        row.update({name: val for name, val in zip(param_list, params)})
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

        # Plot
        x_smooth = np.linspace(min(x), max(x), 500)
        y_smooth = fit_func(x_smooth, *params)

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        plt.scatter(x, y_shift, s=40, alpha=0.6, label="Data", color="#0B75B3")
        plt.plot(x_smooth, y_smooth, label="Fitted Curve", color="#C84747", linewidth=2)

        plt.xlabel("Time (s)", fontdict={'family': 'Arial', 'size': 24, 'weight': 'bold'})
        plt.ylabel("PL Peak Shift (eV)", fontdict={'family': 'Arial', 'size': 24, 'weight': 'bold'})
        ax.tick_params(axis='both', labelsize=18, width=1.5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontname('Arial')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        plt.legend(loc='upper right', frameon=False, prop={'family': 'Arial', 'size': 14})
        fit_text = (
            "y = y0 + A1 * exp(-x / t1)\n"
            f"Eg = {Eg:.4f} eV\n"
            f"R² = {r2:.4f}\n"
            f"y0 = {params[0]:.4f}\n"
            f"A1 = {params[1]:.4f}\n"
            f"t1 = {params[2]:.4f}"
        )
        plt.text(0.5, 0.5, fit_text,
                 transform=ax.transAxes, fontsize=18, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='#C84747', alpha=0.4, edgecolor='black'))

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"fit_{file.replace('.csv', '.png')}"), dpi=600)
        plt.close()

    except Exception as e:
        print(f"  Error fitting {file}: {e}")

# === Save fitting summary ===
results_df['SortKey'] = results_df['File'].str.extract(r'(\d+)', expand=False).astype(int)
results_df = results_df.sort_values('SortKey').drop(columns='SortKey')
results_df.to_csv(os.path.join(output_folder, f"{equation_name}_Fitting_Parameters.csv"), index=False)

print("✅ All done.")
