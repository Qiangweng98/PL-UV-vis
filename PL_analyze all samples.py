import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from datetime import datetime
from matplotlib import cm
import matplotlib.image as mpimg
from numpy.polynomial.polynomial import Polynomial
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap
import glob

# Define directory path for input files and create output directory
#"When you change the input file path, remember there are 3 places have to be changed in this programm #line18 307 326"
input_dir = '/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/250526/Remove noise csv'  # Define the directory path here
output_dir = os.path.join(input_dir, "Output")
os.makedirs(output_dir, exist_ok=True)

# Find all files in the input directory that match the pattern "Nr *.csv"
csv_files = glob.glob(os.path.join(input_dir, "Nr *.csv"))

# Process each CSV file found in the directory
for input_file_path in csv_files:
    # Prepare file-specific output paths
    input_filename = os.path.basename(input_file_path).replace('.csv', '_fitted_results.csv').replace(' ', '_')
    sample_name = os.path.basename(input_file_path).replace('.csv', '')  # Extract sample name (e.g., "Nr 1")
    output_file_path = os.path.join(output_dir, input_filename)

    # Read and preprocess the CSV file containing spectral data.
    data_csv = pd.read_csv(input_file_path, skiprows=4)
    data_csv.rename(columns={data_csv.columns[0]: "Wavelength"}, inplace=True)
    wavelength = data_csv["Wavelength"].values
    cycle_data = data_csv.iloc[:, 1:]
    real_time = cycle_data.columns


    # Define a Gaussian function to be used for curve fitting.
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


    # Use the second derivative and smoothing to approximate the peak position.
    def find_peak_by_2nd_derivative(y, x, window_size=10):
        y_smoothed = uniform_filter1d(y, size=window_size)
        y_2nd_derivative = np.gradient(np.gradient(y_smoothed))
        peak_idx = np.argmin(y_2nd_derivative)
        refined_range = slice(max(0, peak_idx - 5), min(len(y), peak_idx + 6))
        refined_peak_idx = int(refined_range.start + np.argmax(y[refined_range]))
        return x[refined_peak_idx], y[refined_peak_idx]


    # Loop through each cycle’s intensity data and fit peaks to it.
    fitting_results = []

    for i, (cycle_time, intensity) in enumerate(cycle_data.items()):
        smoothed_intensity = uniform_filter1d(intensity, size=10)

        # Define fitting range for each peak (Peak 1: 400-600 nm, Peak 2: 600-950 nm).
        peak1_range = (wavelength >= 400) & (wavelength <= 600)
        peak2_range = (wavelength >= 600) & (wavelength <= 950)
        peak_positions = [np.nan, np.nan]
        peak_intensities = [np.nan, np.nan]
        peak_areas = [np.nan, np.nan]
        peak_fwhms = [np.nan, np.nan]

        # First peak fitting (400-600 nm).
        y_peak1 = smoothed_intensity[peak1_range]
        x_peak1 = wavelength[peak1_range]
        mu_guess1, A_guess1 = find_peak_by_2nd_derivative(y_peak1, x_peak1)
        sigma_guess1 = (max(x_peak1) - min(x_peak1)) / 10

        try:
            popt1, _ = curve_fit(
                gaussian, x_peak1, y_peak1,
                p0=[A_guess1, mu_guess1, sigma_guess1],
                bounds=([0, 400, 0.5], [np.inf, 600, 50]),
                maxfev=10000
            )
            peak_positions[0] = popt1[1]
            peak_intensities[0] = popt1[0]
            peak_areas[0] = trapezoid(gaussian(x_peak1, *popt1), x_peak1)
            peak_fwhms[0] = abs(2.35482 * popt1[2])
        except (RuntimeError, ValueError):
            print(f"Cycle {i + 1}: Failed to fit first peak in range 400-600 nm")

        # Second peak fitting (600-950 nm).
        y_peak2 = smoothed_intensity[peak2_range]
        x_peak2 = wavelength[peak2_range]
        mu_guess2, A_guess2 = find_peak_by_2nd_derivative(y_peak2, x_peak2)
        sigma_guess2 = (max(x_peak2) - min(x_peak2)) / 10

        try:
            popt2, _ = curve_fit(
                gaussian, x_peak2, y_peak2,
                p0=[A_guess2, mu_guess2, sigma_guess2],
                bounds=([0, 600, 0.5], [np.inf, 950, 100]),
                maxfev=10000
            )
            peak_positions[1] = popt2[1]
            peak_intensities[1] = popt2[0]
            peak_areas[1] = trapezoid(gaussian(x_peak2, *popt2), x_peak2)
            peak_fwhms[1] = abs(2.35482 * popt2[2])
        except (RuntimeError, ValueError):
            print(f"Cycle {i + 1}: Failed to fit second peak in range 600-950 nm")

        fitting_results.append([
            i + 1, real_time[i], peak_positions[0], peak_intensities[0], peak_areas[0], peak_fwhms[0],
            peak_positions[1], peak_intensities[1], peak_areas[1], peak_fwhms[1]
        ])

    # Convert fitting results into a DataFrame for structured data handling.
    fitting_df = pd.DataFrame(fitting_results, columns=[
        'Cycle Time', 'Real Time', 'Peak 1 Position', 'Peak 1 Intensity', 'Peak 1 Area', 'Peak 1 FWHM',
        'Peak 2 Position', 'Peak 2 Intensity', 'Peak 2 Area', 'Peak 2 FWHM'
    ])

    # Interpolate missing values in the DataFrame.
    for column in ['Peak 1 Position', 'Peak 1 Intensity', 'Peak 1 Area', 'Peak 1 FWHM',
                   'Peak 2 Position', 'Peak 2 Intensity', 'Peak 2 Area', 'Peak 2 FWHM']:
        nan_indices = fitting_df[column].isna()
        for idx in fitting_df[nan_indices].index:
            prev_idx = fitting_df[~nan_indices & (fitting_df.index < idx)].index.max()
            next_idx = fitting_df[~nan_indices & (fitting_df.index > idx)].index.min()
            if not pd.isna(prev_idx) and not pd.isna(next_idx):
                prev_value = fitting_df.loc[prev_idx, column]
                next_value = fitting_df.loc[next_idx, column]
                fitting_df.loc[idx, column] = prev_value + (next_value - prev_value) * (idx - prev_idx) / (
                            next_idx - prev_idx)

    # Normalize Intensity and Area based on the global maximum.
    global_max_intensity = fitting_df[['Peak 1 Intensity', 'Peak 2 Intensity']].max().max()
    fitting_df['Peak 1 Normalized Intensity'] = fitting_df['Peak 1 Intensity'] / global_max_intensity
    fitting_df['Peak 1 Normalized Area'] = fitting_df['Peak 1 Area'] / fitting_df.loc[0, 'Peak 1 Area']
    fitting_df['Peak 2 Normalized Intensity'] = fitting_df['Peak 2 Intensity'] / global_max_intensity
    fitting_df['Peak 2 Normalized Area'] = fitting_df['Peak 2 Area'] / fitting_df.loc[0, 'Peak 2 Area']

    # Add time column and calculate shift speed and acceleration.
    fitting_df['Time'] = [0] + [(datetime.strptime(real_time[i], '%m/%d/%Y %H:%M:%S') - datetime.strptime(real_time[0],
                                                                                                          '%m/%d/%Y %H:%M:%S')).total_seconds()
                                for i in range(1, len(real_time))]

    fitting_df['Peak 1 Position Shift'] = fitting_df['Peak 1 Position'] - fitting_df['Peak 1 Position'].iloc[0]
    fitting_df['Peak 2 Position Shift'] = fitting_df['Peak 2 Position'] - fitting_df['Peak 2 Position'].iloc[0]

    fitting_df['Peak 1 Shift Speed'] = fitting_df['Peak 1 Position'].diff() / fitting_df['Time'].diff()
    fitting_df['Peak 2 Shift Speed'] = fitting_df['Peak 2 Position'].diff() / fitting_df['Time'].diff()
    fitting_df.loc[0, 'Peak 1 Shift Speed'] = 0  # First cycle speed set to 0
    fitting_df.loc[0, 'Peak 2 Shift Speed'] = 0  # First cycle speed set to 0

    fitting_df['Peak 1 Acceleration'] = fitting_df['Peak 1 Shift Speed'].diff() / fitting_df['Time'].diff()
    fitting_df['Peak 2 Acceleration'] = fitting_df['Peak 2 Shift Speed'].diff() / fitting_df['Time'].diff()
    fitting_df.loc[0, 'Peak 1 Acceleration'] = 0  # First cycle acceleration set to 0
    fitting_df.loc[0, 'Peak 2 Acceleration'] = 0  # First cycle acceleration set to 0

    # Add energy calculation columns.
    fitting_df['Peak 1 Energy(eV)'] = 1239.84 / fitting_df['Peak 1 Position']
    fitting_df['Peak 2 Energy(eV)'] = 1239.84 / fitting_df['Peak 2 Position']

    # Add Peak 1 Area Ratio and Peak 2 Area Ratio to the fitting DataFrame
    fitting_df['Peak 1 Area Ratio'] = fitting_df['Peak 1 Area'] / fitting_df['Peak 1 Area'].iloc[0]
    fitting_df['Peak 2 Area Ratio'] = fitting_df['Peak 2 Area'] / fitting_df['Peak 2 Area'].iloc[0]

    # Rearrange columns to insert the new columns after the respective "Peak 1 Area" and "Peak 2 Area"
    columns_order = [
        'Cycle Time', 'Real Time', 'Time',
        'Peak 1 Position', 'Peak 1 Position Shift', 'Peak 1 Shift Speed', 'Peak 1 Acceleration',
        'Peak 1 Intensity', 'Peak 1 Normalized Intensity', 'Peak 1 Area', 'Peak 1 Area Ratio', 'Peak 1 Normalized Area',
        'Peak 1 FWHM', 'Peak 1 Energy(eV)',
        'Peak 2 Position', 'Peak 2 Position Shift', 'Peak 2 Shift Speed', 'Peak 2 Acceleration',
        'Peak 2 Intensity', 'Peak 2 Normalized Intensity', 'Peak 2 Area', 'Peak 2 Area Ratio', 'Peak 2 Normalized Area',
        'Peak 2 FWHM', 'Peak 2 Energy(eV)'
    ]
    fitting_df = fitting_df[columns_order]

    # Save the modified DataFrame to CSV
    output_dir = os.path.join(input_dir, "Output")
    output_fitted_results_dir = os.path.join(output_dir, "Fitted results")
    os.makedirs(output_fitted_results_dir, exist_ok=True)
    output_file_path = os.path.join(output_fitted_results_dir, input_filename)
    fitting_df.to_csv(output_file_path, index=False)
    print(f"Fitting results saved to {output_file_path}")

    # Colors provided (lightest to darkest)
    colors = [
        "#012A61", "#015696", "#0B75B3", "#4596CD", "#89CAEA",
        "#BBE6FA", "#E2F4FE", "#F1EEED", "#FCCDC9", "#EE9D9F",
        "#DE6A69", "#C84747", "#982B2D"
    ]

    # Generate colormap
    color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)

    # Visualization for Plot 1
    plt.figure(figsize=(12, 8))
    selected_indices = [0, 1, 4, 9, 19, 49, 89]

    # Generate colors from the colormap
    selected_colors = [color_map(i / (len(selected_indices) - 1)) for i in range(len(selected_indices))]
    labels = ["Cycle 1", "Cycle 2", "Cycle 5", "Cycle 10", "Cycle 20", "Cycle 50", "Cycle 90"]

    # Calculate maximum y value for selected cycles
    y_max = 0
    for idx in selected_indices:
        y_max = max(y_max, cycle_data.iloc[:, idx].max())
    y_max *= 1.1

    for idx, color, label in zip(selected_indices, selected_colors, labels):
        intensity = cycle_data.iloc[:, idx]
        plt.plot(wavelength, intensity, color=color, label=label, linewidth=1.5)

        # Plot Peak 1 and Peak 2 fits for each cycle
        if fitting_df.loc[idx, 'Peak 1 Position'] > 0:
            peak1_fit_x = wavelength[(wavelength >= 400) & (wavelength <= 600)]
            peak1_fit_y = gaussian(peak1_fit_x, fitting_df.loc[idx, 'Peak 1 Intensity'],
                                   fitting_df.loc[idx, 'Peak 1 Position'],
                                   fitting_df.loc[idx, 'Peak 1 FWHM'] / 2.35482)
            plt.plot(peak1_fit_x, peak1_fit_y, linestyle='--', color=color, alpha=0.7)

        if fitting_df.loc[idx, 'Peak 2 Position'] > 0:
            peak2_fit_x = wavelength[(wavelength >= 600) & (wavelength <= 950)]
            peak2_fit_y = gaussian(peak2_fit_x, fitting_df.loc[idx, 'Peak 2 Intensity'],
                                   fitting_df.loc[idx, 'Peak 2 Position'],
                                   fitting_df.loc[idx, 'Peak 2 FWHM'] / 2.35482)
            plt.plot(peak2_fit_x, peak2_fit_y, linestyle='--', color=color, alpha=0.7)

    # Formatting changes
    plt.xlim(510, 940)
    y_min = -y_max / 40
    plt.ylim(y_min, y_max)
    plt.yticks([])  # Hide y-axis tick labels
    plt.xlabel("Wavelength (nm)", fontsize=32, weight='bold', labelpad=10)
    plt.ylabel("PL Intensity (a.u.)", fontsize=32, weight='bold', labelpad=10)
    plt.xticks(fontsize=24, weight='bold')
    plt.yticks(fontsize=24, weight='bold')
    plt.tick_params(axis='both', which='both', direction='in', width=2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    # plt.title(f"PL Spectra with Gaussian Fit Peaks - {sample_name}", fontsize=32, weight='bold', fontname='Arial', pad=20)

    # Add legend showing only the solid lines corresponding to cycles
    solid_line_handles = []
    solid_line_labels = []

    for idx, color, label in zip(selected_indices, selected_colors, labels):
        handle, = plt.plot([], [], color=color, label=label, linewidth=1.5)
        solid_line_handles.append(handle)
        solid_line_labels.append(label)

    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.legend(solid_line_handles, solid_line_labels, frameon=False, loc='upper right',
               prop={'family': 'Arial', 'weight': 'bold', 'size': 24})

    # Save Plot 1
    output_image_path = os.path.join(output_dir, "Fitted curves", f"{sample_name}_fit_plot.png")
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)  # Ensure the directory exists
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_image_path}")
    plt.close()

    # Visualization for Plot 2
    plt.figure(figsize=(12, 8))

    # Colors provided (lightest to darkest)
    colors = [
        "#012A61", "#015696", "#0B75B3", "#4596CD", "#89CAEA",
        "#BBE6FA", "#E2F4FE", "#F1EEED", "#FCCDC9", "#EE9D9F",
        "#DE6A69", "#C84747", "#982B2D"
    ]

    # Generate colormap
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)

    # Normalize the color range from 1 to 90 (number of cycles)
    norm = Normalize(vmin=0, vmax=600)

    # Plot all 90 cycles with the new colormap
    for i, column in enumerate(cycle_data.columns):
        intensity = cycle_data[column].values
        plt.plot(wavelength, intensity, color=color_map(i / (len(cycle_data.columns) - 1)), linewidth=1.5)

    # Set axis limits and labels
    plt.xlim(510, 940)
    y_min = -y_max / 40
    plt.ylim(y_min, y_max)
    plt.yticks([])  # Hide y-axis tick labels
    plt.tick_params(axis='y', which='both', left=False, right=False)  # Disable left and right ticks
    plt.xlabel("Wavelength (nm)", fontsize=32, weight='bold', labelpad=10)
    plt.ylabel("PL Intensity (a.u.)", fontsize=32, weight='bold', labelpad=10)
    plt.xticks(fontsize=24, weight='bold', fontname='Arial')
    plt.tick_params(axis='both', which='both', direction='in', width=2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)

    # Add the image near the top-left corner
    photo_dir = "/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/250526/PVK Photos"
    image_path_jpg = os.path.join(photo_dir, f"{sample_name}.jpg")
    image_path_jpeg = os.path.join(photo_dir, f"{sample_name}.jpeg")

    if os.path.exists(image_path_jpg):
        img = mpimg.imread(image_path_jpg)
        ax_img = plt.gcf().add_axes([0.048, 0.625, 0.28, 0.28])  # Adjust [x, y, width, height] for position and size
        ax_img.imshow(img)
        ax_img.axis('off')
    elif os.path.exists(image_path_jpeg):
        img = mpimg.imread(image_path_jpeg)
        ax_img = plt.gcf().add_axes([0.048, 0.625, 0.28, 0.28])  # Adjust [x, y, width, height] for position and size
        ax_img.imshow(img)
        ax_img.axis('off')
    else:
        print(f"Image not found for {sample_name} in {photo_dir}")

    # Add chemical formula as text in the top-left corner
    # Read the recipe CSV file
    recipe_path = "/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/250526/Formatted Recipes.csv"
    recipe_data = pd.read_csv(recipe_path)
    sample_number = int(sample_name.split(' ')[1])  # Extract the number part from "Nr 29"
    chemical_formula = recipe_data.loc[recipe_data['Nr.'] == sample_number, 'Recipe'].values[0]


    # Convert the chemical formula to a LaTeX-friendly format with subscripts
    def format_formula_with_correct_subscripts(formula):
        formatted = ""
        i = 0
        while i < len(formula):
            if formula[i].isalpha():  # If it's an alphabetic character (element symbol)
                formatted += formula[i]
            elif formula[i].isdigit() or formula[i] == '.':  # If it's part of a number
                number = ""
                while i < len(formula) and (formula[i].isdigit() or formula[i] == '.'):
                    number += formula[i]
                    i += 1
                formatted += f"_{{{number}}}"  # Wrap the full number in LaTeX subscript
                continue  # Skip the increment as it's handled in the inner loop
            else:
                formatted += formula[i]
            i += 1
        return f"${formatted}$"  # Ensure the formula is wrapped for LaTeX rendering


    formatted_formula = format_formula_with_correct_subscripts(chemical_formula)

    # Add the formatted chemical formula to the plot at the top-left corner
    plt.text(0.01, 1.22, formatted_formula, transform=plt.gca().transAxes,
             fontsize=24, fontname='Arial', weight='bold', va='top', ha='left', zorder=10)

    # Add "Shift x nm" below the image
    shift_x_nm = round(fitting_df.loc[fitting_df['Cycle Time'] == 83, 'Peak 2 Position'].values[0] -
                       fitting_df.loc[fitting_df['Cycle Time'] == 1, 'Peak 2 Position'].values[0])
    plt.text(0.88, 0.86, f"Shift\n{shift_x_nm} nm", transform=plt.gcf().transFigure,
             fontsize=24, fontname='Arial', weight='bold', ha='center', va='center')

    # Add a colorbar to indicate time in seconds (from 0 to 600s)
    sm = cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array(np.linspace(0, 600, 90))  # Explicitly map 90 cycles to the range 0-600s

    cbar = plt.colorbar(sm, orientation="vertical", pad=0.04, ax=plt.gca(), aspect=12)
    cbar.set_label('', fontsize=24, weight='bold', labelpad=0)  # Remove colorbar label
    cbar.set_ticks([0, 600])
    cbar.set_ticklabels(['0s', '600s'])
    cbar.ax.tick_params(labelsize=24, width=2)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_weight('bold')

    # Save Plot 2
    output_image_path_2 = os.path.join(output_dir, "Original curves", f"{sample_name}_all_cycles_plot.png")
    os.makedirs(os.path.dirname(output_image_path_2), exist_ok=True)  # Ensure the directory exists
    plt.tight_layout()
    plt.savefig(output_image_path_2, dpi=300, bbox_inches="tight")
    print(f"Plot (All Cycles) saved to {output_image_path_2}")
    plt.close()

    # Visualization for Plot 3
    # Normalize the intensity for each cycle
    normalized_cycle_data = cycle_data.divide(cycle_data.max(axis=0), axis=1)

    # Define the Wavelength range (630 to 870 nm)
    wavelength_mask = (wavelength >= 630) & (wavelength <= 870)
    filtered_wavelengths = wavelength[wavelength_mask]
    filtered_energy = 1239.84 / filtered_wavelengths
    filtered_normalized_data = normalized_cycle_data[wavelength_mask]

    # Colors provided (lightest to darkest)
    colors = [
        "#012A61", "#015696", "#0B75B3", "#4596CD", "#89CAEA",
        "#BBE6FA", "#E2F4FE", "#F1EEED", "#FCCDC9", "#EE9D9F",
        "#DE6A69", "#C84747", "#982B2D"
    ]

    # Generate colormap
    color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = plt.imshow(
        filtered_normalized_data.T,  # Transpose to align cycles vertically
        aspect='auto',
        cmap=color_map,  # Use the new custom colormap
        extent=[filtered_wavelengths.min(), filtered_wavelengths.max(), 0, 600],
        origin='lower'
    )

    # Calculate the positions where Normalized PL Intensity = 1 for each cycle
    cycle_max_positions = [
        filtered_wavelengths[np.argmax(filtered_normalized_data.T.iloc[i] == 1)]
        for i in range(filtered_normalized_data.T.shape[0])
    ]
    cycle_indices = np.linspace(0, 600, len(cycle_max_positions))  # Corresponding time for each cycle

    # Add black circular markers for points where Normalized PL Intensity = 1
    for x, y in zip(cycle_max_positions, cycle_indices):
        plt.plot(
            x, y, 'o', markersize=8, markeredgewidth=1.5, markeredgecolor='black', markerfacecolor=(0, 0, 0, 0.2)
        )  # Internal fill is 20%, external black border

    # Perform a 8th-order polynomial fit
    poly_coeffs = np.polyfit(cycle_indices, cycle_max_positions, 8)
    poly_fit = np.poly1d(poly_coeffs)

    # Generate smooth line from the polynomial fit
    fit_x = poly_fit(cycle_indices)

    # Plot the polynomial fit line
    plt.plot(fit_x, cycle_indices, color='black', linewidth=3, alpha=1)

    # Lower x-axis (Wavelength)
    plt.xlabel("Wavelength (nm)", fontsize=32, weight='bold', labelpad=10)

    # Y-axis (Time in seconds)
    plt.ylabel("Time (s)", fontsize=32, weight='bold', labelpad=10)
    plt.yticks(
        ticks=np.linspace(0, 600, 7),
        labels=[int(val) for val in np.linspace(0, 600, 7)],
        fontsize=24, weight='bold'
    )

    # Format axes appearance
    plt.xticks(fontsize=24, weight='bold')
    plt.tick_params(axis='both', which='both', direction='in', width=2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)

    # Upper x-axis (Energy in eV)
    top_ax = plt.gca().twiny()
    top_ax.set_xlim(filtered_wavelengths.min(), filtered_wavelengths.max())
    energy_ticks = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]  # Potential Energy ticks
    energy_tick_positions = [1239.84 / e for e in energy_ticks if 1239.84 / e >= 630 and 1239.84 / e <= 870]
    energy_tick_labels = [f"{1239.84 / pos:.1f}" for pos in energy_tick_positions]
    top_ax.set_xticks(energy_tick_positions)
    top_ax.set_xticklabels(energy_tick_labels, fontsize=24, weight='bold')
    top_ax.set_xlabel("Energy (eV)", fontsize=32, weight='bold', labelpad=10)
    top_ax.tick_params(axis='x', direction='in', width=2)

    # Add a colorbar
    cbar = plt.colorbar(heatmap, orientation="vertical", aspect=12, pad=0.02)
    cbar.set_label('Normalized PL Intensity', fontsize=24, weight='bold', labelpad=20)
    cbar.ax.tick_params(labelsize=24, width=1)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_weight('bold')

    # Save Plot 3
    output_heatmap_path = os.path.join(output_dir, "Heatmaps", f"{sample_name}_heatmap.png")
    os.makedirs(os.path.dirname(output_heatmap_path), exist_ok=True)
    plt.savefig(output_heatmap_path, dpi=300,
                bbox_inches="tight")  # Save with tight bounding box for consistent whitespace
    print(f"Heatmap saved to {output_heatmap_path}")
    plt.close()
