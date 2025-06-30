import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# Define file paths
input_file_path = '/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/241211/Nr 8.csv'
input_dir = os.path.dirname(input_file_path)
input_filename = os.path.basename(input_file_path)
output_dir = os.path.join(input_dir, "Remove noise csv")
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(output_dir, input_filename)

# Read the CSV file
with open(input_file_path, 'r') as f:
    header_lines = [next(f) for _ in range(5)]  # Retain the first five header lines

data_csv = pd.read_csv(input_file_path, skiprows=5)
data_csv.rename(columns={data_csv.columns[0]: "Wavelength"}, inplace=True)
wavelength = data_csv["Wavelength"].values
cycle_data = data_csv.iloc[:, 1:]

# Replace all negative intensity values with 0
cycle_data[cycle_data < 0] = 0

# Define the noise filtering logic
threshold = 50  # Noise threshold for the full range
corrected_data = cycle_data.copy()

for cycle in range(cycle_data.shape[1]):  # Iterate over each column (cycle)
    current_cycle = cycle_data.iloc[:, cycle].values
    corrected_cycle = current_cycle.copy()

    idx = 0
    while idx < len(current_cycle):
        # Check for both upward spikes and downward dips
        if (
            idx > 0 and
            (abs(current_cycle[idx] - current_cycle[idx - 1]) > threshold or
             (current_cycle[idx - 1] - current_cycle[idx]) > threshold)
        ):
            # Mark as noise and perform interpolation
            start_idx = idx - 1
            end_idx = idx + 1

            while end_idx < len(current_cycle) and (
                abs(current_cycle[end_idx] - current_cycle[end_idx - 1]) > threshold or
                (current_cycle[end_idx - 1] - current_cycle[end_idx]) > threshold
            ):
                end_idx += 1

            if start_idx >= 0 and end_idx < len(current_cycle):
                start_value = current_cycle[start_idx]
                end_value = current_cycle[end_idx]
                interpolated_values = np.linspace(start_value, end_value, end_idx - start_idx + 1)[1:-1]
                corrected_cycle[start_idx + 1:end_idx] = interpolated_values

            idx = end_idx  # Skip the processed part
        else:
            idx += 1

    # Second pass to handle new noise introduced by replacing negative values
    idx = 0
    while idx < len(corrected_cycle):
        if (
            idx > 0 and
            (abs(corrected_cycle[idx] - corrected_cycle[idx - 1]) > threshold or
             (corrected_cycle[idx - 1] - corrected_cycle[idx]) > threshold)
        ):
            # Mark as noise and perform interpolation
            start_idx = idx - 1
            end_idx = idx + 1

            while end_idx < len(corrected_cycle) and (
                abs(corrected_cycle[end_idx] - corrected_cycle[end_idx - 1]) > threshold or
                (corrected_cycle[end_idx - 1] - corrected_cycle[end_idx]) > threshold
            ):
                end_idx += 1

            if start_idx >= 0 and end_idx < len(corrected_cycle):
                start_value = corrected_cycle[start_idx]
                end_value = corrected_cycle[end_idx]
                interpolated_values = np.linspace(start_value, end_value, end_idx - start_idx + 1)[1:-1]
                corrected_cycle[start_idx + 1:end_idx] = interpolated_values

            idx = end_idx  # Skip the processed part
        else:
            idx += 1

    corrected_data.iloc[:, cycle] = corrected_cycle

# Save the corrected data while retaining the header lines
with open(output_file_path, 'w') as f:
    f.writelines(header_lines)  # Write the original header lines

corrected_data.insert(0, "Wavelength", wavelength)  # Only correct the middle data, retain the original first column
corrected_data.to_csv(output_file_path, index=False, header=False, mode='a')  # Append the corrected data to the file

# Print confirmation message
print(f"Corrected CSV file has been successfully exported to: {output_file_path}")
