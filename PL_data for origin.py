import os
import pandas as pd
import re

# Define the input directory, x, and y at the beginning for easy modification
input_directory = "/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/241211/Remove noise csv/Output/Fitted results"  # Change this path as needed
output_directory = os.path.join(input_directory, "For Origin")
os.makedirs(output_directory, exist_ok=True)

# Map column letters to names for file naming
column_names = {
    'A': "Cycle Time", 'B': "Real Time", 'C': "Time", 'D': "Peak 1 Position", 'E': "Peak 1 Position Shift",
    'F': "Peak 1 Acceleration", 'G': "Peak 1 Intensity", 'H': "Peak 1 Normalized Intensity",
    'I': "Peak 1 Energy (eV) Calculated from Position", 'J': "Peak 1 Area", 'K': "Peak 1 Area Ratio",
    'L': "Peak 1 Normalized Area", 'M': "Peak 1 FWHM", 'N': "Peak 1 Energy (eV)", 'O': "Peak 2 Position",
    'P': "Peak 2 Position Shift", 'Q': "Peak 2 Shift Speed", 'R': "Peak 2 Acceleration",
    'S': "Peak 2 Intensity", 'T': "Peak 2 Normalized Intensity", 'U': "Peak 2 Area",
    'V': "Peak 2 Area Ratio", 'W': "Peak 2 Normalized Area", 'X': "Peak 2 FWHM", 'Y': "Peak 2 Energy (eV) Calculated from Positio"
}

x = 'C'  # Define x here, options: 'A' or 'C'
y = 'V'  # Define y here, based on your column letters

# Generate file name based on x and y
output_file_name = f"{column_names[x]}_{column_names[y]}.csv"
output_file_path = os.path.join(output_directory, output_file_name)

# Collect all CSV files in the input directory and sort by numeric value in filename
csv_files = sorted(
    [f for f in os.listdir(input_directory) if re.match(r"Nr_\d+_fitted_results\.csv", f)],  # Match 'Nr_1_fitted_results.csv', etc.
    key=lambda name: int(re.search(r"\d+", name).group())  # Extract and sort by numeric part of the filename
)

# Verify that there are files to process
if not csv_files:
    print("No CSV files found in the input directory.")
else:
    # Read the first CSV to get the first column data based on x
    first_file_path = os.path.join(input_directory, csv_files[0])
    first_df = pd.read_csv(first_file_path)

    # Select the appropriate first column based on x ('A' or 'C')
    x_column_data = first_df[column_names[x]]

    # Prepare data for each subsequent column based on y
    y_column_data = [x_column_data]  # Initialize with the first column data
    file_numbers = []  # Store the file number for each y column

    for file in csv_files:
        file_path = os.path.join(input_directory, file)
        df = pd.read_csv(file_path)

        # Add the selected y column to the list
        y_column_data.append(df[column_names[y]])

        # Extract file number from filename, e.g., 'Nr_6_fitted_results.csv' -> 'Nr 6'
        file_number = "Nr " + re.search(r"\d+", file).group()
        file_numbers.append(file_number)

    # Combine all columns into a single DataFrame
    output_df = pd.concat(y_column_data, axis=1)

    # Set the column headers with the file identifiers, starting with the x column
    output_df.columns = [column_names[x]] + file_numbers

    # Save the output DataFrame to CSV
    output_df.to_csv(output_file_path, index=False)
    print(f"Output saved to: {output_file_path}")
