import os
import pandas as pd
import numpy as np
import re

# Define input and output directories
input_directory = "/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/250416/Remove noise csv/Output/Fitted results"
output_directory = os.path.join(input_directory, "For Origin")
os.makedirs(output_directory, exist_ok=True)

# Collect all matching CSV files and extract their number X
csv_files = sorted(
    [f for f in os.listdir(input_directory) if re.match(r"Nr_\d+_fitted_results\.csv", f)],
    key=lambda name: int(re.search(r"\d+", name).group())  # Sort by numeric order
)

# Extract IDs from file names
ids = [re.search(r"Nr_(\d+)_fitted_results\.csv", f).group(1) for f in csv_files]

# Initialize output DataFrame
output_data = pd.DataFrame(columns=["A", "B", "C", "D", "E"])
output_data.loc[0] = [np.nan, "eV", "Peak Position Shift at 10s", "Peak Position Shift at 200s", "Peak Position Shift at 600s"]

# Loop over files and extract required values
for i, (file_name, id_str) in enumerate(zip(csv_files, ids)):
    file_path = os.path.join(input_directory, file_name)
    data = pd.read_csv(file_path)

    # Fill row with data
    output_data.loc[i + 1, "A"] = f"Nr {id_str}"
    output_data.loc[i + 1, "B"] = data.iloc[0, 24]
    output_data.loc[i + 1, "C"] = data.iloc[1, 15]
    output_data.loc[i + 1, "D"] = data.iloc[28, 15]
    output_data.loc[i + 1, "E"] = data.iloc[83, 15]

# Save result
output_file_path = os.path.join(output_directory, "Band gap_Peak 2 Position Shift.csv")
output_data.to_csv(output_file_path, index=False, header=False)

print(f"Output saved to: {output_file_path}")
