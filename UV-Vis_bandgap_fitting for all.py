from typing import Union, Any
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter1d

# Input CSV file paths
uv_data_path = "/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/250416/Remove noise csv/UV data.csv"
eg_csv_path = "/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/250416/Remove noise csv/Output/Eg fomulation calculated.csv"

# Load UV data and Eg values
df = pd.read_csv(uv_data_path)  # UV data containing wavelength and absorbance
eg_df = pd.read_csv(eg_csv_path)  # Eg data with sample numbers and calculated Eg values

# Create output directories for results
output_dir = os.path.join(os.path.dirname(uv_data_path), "Output")
uv_vis_fig_dir = os.path.join(output_dir, "UV Vis Fig")
os.makedirs(uv_vis_fig_dir, exist_ok=True)

# Path for output CSV file
output_csv_path = os.path.join(output_dir, "UV-Vis-Eg and Eu.csv")

# Initialize a list to store results for all samples (Eg and Eu values)
results = []

# Initialize or load the output CSV file
if os.path.exists(output_csv_path):
    # If the output file already exists, load it
    uv_vis_eg_df = pd.read_csv(output_csv_path)
else:
    # If the output file doesn't exist, initialize a new DataFrame
    uv_vis_eg_df = pd.DataFrame({
        "Sample Nr": eg_df["Sample Nr."],  # Use sample numbers from Eg CSV
        "UV-Vis calculated Eg (eV)": np.nan,  # Initialize Eg with NaN
        "UV-Vis calculated Eu (meV)": np.nan  # Initialize Eu with NaN
    })

# Iterate through all sample numbers from the first column of the Eg CSV file
for sample_number in eg_df["Sample Nr."]:
    try:
        # Check if the sample number exists in the UV data
        if str(sample_number) not in df.columns:
            print(f"Sample number {sample_number} not found in UV data. Skipping.")
            continue

        # Extract Eg for the current sample from the Eg CSV file
        eg_row = eg_df.loc[eg_df["Sample Nr."] == sample_number]
        if eg_row.empty:
            print(f"Eg value for sample number {sample_number} not found. Skipping.")
            continue

        Eg_estimate = eg_row["Fomulation calculated Eg (eV)"].values[0]

        # Extract wavelength and sample data
        wavelength = df['Wavel.'].values.astype(float)  # Extract wavelength from the first column
        absorbance = df[str(sample_number)].values.astype(float)  # Extract absorbance for the specified sample

        # Convert wavelength to photon energy (eV)
        h_v = 1240 / wavelength  # Photon energy
        A = absorbance  # Absorbance
        absorption_coefficient_hv_square = (A * h_v) ** 2  # Calculate (Ahv)²

        # Define the range for the first derivative maximum based on Eg_estimate
        derivative_range = (h_v >= Eg_estimate - 0.02) & (h_v <= Eg_estimate + 0.1)  # Updated range
        h_v_derivative = h_v[derivative_range]
        absorption_coefficient_hv_square_derivative = absorption_coefficient_hv_square[derivative_range]

        # Check if derivative_range contains valid data
        if len(h_v_derivative) == 0 or len(absorption_coefficient_hv_square_derivative) == 0:
            raise ValueError(f"No data found in the derivative range for sample number {sample_number}.")

        # Apply Gaussian smoothing and calculate the first derivative
        smoothed_curve = gaussian_filter1d(absorption_coefficient_hv_square_derivative, sigma=2)
        diff_y = np.gradient(smoothed_curve, h_v_derivative)
        max_diff_index = np.argmax(diff_y)
        max_diff_x = h_v_derivative[max_diff_index]  # x value for the maximum derivative
        max_diff_y = absorption_coefficient_hv_square_derivative[max_diff_index]  # y value for the maximum derivative

        # Baseline calculation
        baseline_left = Eg_estimate - 0.15
        baseline_right = Eg_estimate - 0.08
        baseline_range = (h_v >= baseline_left) & (h_v <= baseline_right)
        baseline_x = h_v[baseline_range]
        baseline_y = absorption_coefficient_hv_square[baseline_range]

        # Check if there are enough points in the baseline range
        if len(baseline_x) < 3:
            raise ValueError(
                f"Not enough points to fit the baseline in range {baseline_left:.2f} to {baseline_right:.2f} eV.")

        # Select 3 evenly spaced points within the baseline range
        selected_indices = np.linspace(0, len(baseline_x) - 1, 3, dtype=int)
        baseline_x_selected = baseline_x[selected_indices]
        baseline_y_selected = baseline_y[selected_indices]

        # Fit a line through the selected baseline points
        baseline_coefficients = np.polyfit(baseline_x_selected, baseline_y_selected, 1)
        baseline_a, baseline_b = baseline_coefficients

        # Define baseline plot range
        baseline_plot_start = min(baseline_x_selected)
        baseline_plot_end = max_diff_x
        baseline_plot_range = (h_v >= baseline_plot_start) & (h_v <= baseline_plot_end)
        baseline_fit_line = baseline_a * h_v[baseline_plot_range] + baseline_b

        # Fitted line calculation
        tangent_range = (h_v >= max_diff_x - 0.02) & (h_v <= max_diff_x + 0.02)
        tangent_x = h_v[tangent_range]
        tangent_y = absorption_coefficient_hv_square[tangent_range]

        # Check if there are enough points in the tangent range
        if len(tangent_x) < 5:
            raise ValueError(
                f"Not enough points to fit the tangent line in range {max_diff_x - 0.02:.2f} to {max_diff_x + 0.02:.2f} eV.")

        # Fit a line through the selected tangent points
        tangent_coefficients = np.polyfit(tangent_x, tangent_y, 1)
        tangent_slope, tangent_intercept = tangent_coefficients
        fitted_line = tangent_slope * h_v + tangent_intercept

        # Dynamically calculate x-axis range based on Eg_estimate
        x_left = Eg_estimate - 0.15
        x_right = Eg_estimate + 0.13

        # Dynamically calculate y-axis range based on the updated x-axis range
        x_axis_range = (h_v >= x_left) & (h_v <= x_right)  # Update x-axis range
        y_min = 0  # Fix y-axis minimum value at 0
        y_max = np.max(
            absorption_coefficient_hv_square[x_axis_range]) * 1.1  # Dynamically calculate y-axis maximum value

        # Determine y-axis tick interval based on y_max and select 4-6 ticks
        if y_max > 1:
            y_tick_interval = 1.0
        elif y_max > 0.1:
            y_tick_interval = 0.2
        else:
            y_tick_interval = 0.05

        # Generate y-axis ticks with a maximum of 5 ticks
        y_ticks = np.arange(0, y_max + y_tick_interval, y_tick_interval)
        if len(y_ticks) > 5:
            y_ticks = y_ticks[::len(y_ticks) // 5]  # Reduce ticks to a maximum of 5

        # Plot the data
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the original curve
        ax.plot(h_v, absorption_coefficient_hv_square, label="(A$\mathbf{h\\nu}$)²", color='#015696', linewidth=2)

        # Plot the baseline (only in the defined range)
        ax.plot(h_v[baseline_plot_range], baseline_fit_line, '--', label="Baseline", color='#89CAEA', linewidth=2)

        # Plot the tangent line (only in the defined range)
        ax.plot(h_v, fitted_line, '--', label="Fitted line", color='#89CAEA', linewidth=2)

        # Highlight the intersection point (without displaying it in the legend)
        intersection_x = (baseline_b - tangent_intercept) / (tangent_slope - baseline_a)
        intersection_y = tangent_slope * intersection_x + tangent_intercept  # y value at the intersection
        Eg = round(intersection_x, 2)  # Bandgap value rounded to 2 decimal places
        ax.plot(
            intersection_x, intersection_y, 'D',
            markersize=8, markeredgewidth=1, markeredgecolor='#4596CD',
            markerfacecolor=(0, 0, 0, 0.8), zorder=5
        )

        # Dynamically adjust left y-axis range
        y_data_max = np.max(absorption_coefficient_hv_square[x_axis_range])  # Get max of (Ahv)^2 in x-axis range
        y_min = 0

        # Adjust y_max dynamically within 1.5x to 1.8x range
        y_max_base = y_data_max * 1.6
        y_max_adjusted = None
        for factor in [1.5, 1.6, 1.7, 1.8]:
            candidate = y_data_max * factor
            if candidate >= y_max_base:
                y_max_adjusted = candidate
                break

        # Find a suitable y_max that matches the standard tick multiples
        possible_multiples = [0.002, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
        y_max_candidates = [multiple for multiple in possible_multiples if multiple >= y_max_adjusted]
        y_max = min(y_max_candidates, default=y_max_base)

        # Determine a suitable tick interval for y-axis
        tick_intervals = [0.002, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
        max_ticks = 6  # Maximum number of ticks allowed
        min_ticks = 4  # Minimum number of ticks allowed

        # Adjust the tick interval dynamically
        for interval in tick_intervals:
            if y_max / interval <= max_ticks and y_max / interval >= min_ticks:
                y_tick_interval = interval
                break
        else:
            y_tick_interval = tick_intervals[-1]  # Default to the smallest interval if no match

        # Generate y-axis ticks
        y_ticks = np.arange(0, y_max + y_tick_interval, y_tick_interval)

        # Automatically determine decimal precision based on tick interval
        if y_tick_interval >= 1:
            y_tick_format = lambda x: f"{int(x)}"  # No decimals
        elif y_tick_interval >= 0.1:
            y_tick_format = lambda x: f"{x:.1f}"  # One decimal
        elif y_tick_interval >= 0.01:
            y_tick_format = lambda x: f"{x:.2f}"  # Two decimals
        else:
            y_tick_format = lambda x: f"{x:.3f}"  # Three decimals or more

        # Set y-axis range and ticks
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([y_tick_format(tick) for tick in y_ticks], fontsize=24, fontname="Arial", weight="bold")

        # Format the axes and labels
        ax.set_xlim(x_left, x_right)

        # Set custom tick intervals for x-axis
        x_ticks = np.arange(np.ceil(x_left * 10) / 10, np.floor(x_right * 10) / 10 + 0.1, 0.1)  # Steps of 0.1
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{tick:.1f}" for tick in x_ticks], fontsize=24, fontname="Arial", weight="bold")

        # Set axis labels with Arial font and bold weight
        ax.set_xlabel("$\mathbf{h\\nu}$ (eV)", fontsize=32, weight='bold', labelpad=10, fontname="Arial", color='black')
        ax.set_ylabel("(A$\mathbf{h\\nu}$)²", fontsize=32, weight='bold', labelpad=10, fontname="Arial",
                      color='#015696')

        # Customize ticks and set bold Arial font for tick labels
        ax.tick_params(axis='x', which='both', direction='in', width=2, length=6, labelsize=24,
                       labelcolor='black')  # x-axis ticks
        ax.tick_params(axis='y', which='both', direction='in', width=2, length=6, labelsize=24,
                       labelcolor='#015696')  # y-axis ticks

        # Apply custom font and weight to tick labels
        for label in ax.get_xticklabels():
            label.set_fontname("Arial")
            label.set_weight("bold")
            label.set_color('black')  # x-axis tick label color

        for label in ax.get_yticklabels():
            label.set_fontname("Arial")
            label.set_weight("bold")
            label.set_color('#015696')  # y-axis tick label color

        # Set bold line widths for axes and change left y-axis color
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            if spine.spine_type in ['left']:
                spine.set_color('#015696')

        # right part 2
        # Calculate ln(A) for the secondary y-axis
        A = absorbance  # Ensure A is defined based on your data
        ln_A = np.log(A)

        # Dynamically calculate y-axis range for ln(A)
        ln_A_x_axis_range = (h_v >= x_left) & (h_v <= x_right)
        ln_A_min = np.min(ln_A[ln_A_x_axis_range]) * 1.2
        ln_A_max = np.max(ln_A[ln_A_x_axis_range])

        # Adjust the range slightly for padding
        ln_A_range_padding = max((ln_A_max - ln_A_min) * 0.1, 0.2)
        ln_A_min -= ln_A_range_padding
        ln_A_max += ln_A_range_padding

        # Set custom tick intervals for the right y-axis (4–6 ticks, properly formatted)
        ln_A_tick_interval = max(0.1, (ln_A_max - ln_A_min) / 5)  # Determine a reasonable interval
        ln_A_tick_interval = np.round(ln_A_tick_interval, 1) if ln_A_tick_interval < 1 else np.ceil(ln_A_tick_interval)
        ln_A_ticks = np.arange(np.floor(ln_A_min / ln_A_tick_interval) * ln_A_tick_interval,
                               ln_A_max + ln_A_tick_interval, ln_A_tick_interval)

        ax2 = ax.twinx()
        ax2.plot(h_v, ln_A, label="ln(A)", color="#C84747", linewidth=2)  # Plot ln(A) with a red solid line
        ax2.set_ylabel("ln(A)", fontsize=32, weight='bold', labelpad=10, fontname="Arial", color="#C84747")
        ax2.tick_params(axis='y', which='both', direction='in', width=2, length=6, labelsize=24, labelcolor="#C84747")
        ax2.spines["right"].set_color("#C84747")  # Set the color of the right y-axis
        for spine in ["top", "right"]:
            ax2.spines[spine].set_linewidth(2)

        # Customize y-ticks for ln(A)
        ax2.set_yticks(ln_A_ticks)
        ax2.set_yticklabels([f"{int(tick) if tick.is_integer() else tick:.1f}" for tick in ln_A_ticks], fontsize=24,
                            fontname="Arial", weight="bold")
        ax2.set_ylim(ln_A_min, ln_A_max)  # Dynamically set the y-axis range for ln(A)

        # Define the range for ln(A) and hv
        ln_A_tangent_window = (h_v >= x_left + 0.1) & (h_v <= x_right - 0.1)

        # Calculate the first derivative of ln(A) with respect to h_v
        ln_A_diff = np.gradient(ln_A, h_v)

        # Find the index of the maximum derivative within the specified range
        ln_A_tangent_index = np.argmax(np.abs(ln_A_diff[ln_A_tangent_window])) + np.where(ln_A_tangent_window)[0][0]

        # Define the x range for the tangent fit (around the derivative maximum)
        tangent_fit_window = (h_v >= h_v[ln_A_tangent_index] - 0.03) & (h_v <= h_v[ln_A_tangent_index] + 0.02)

        # Get the x and y values for the tangent fit
        tangent_fit_x = h_v[tangent_fit_window]
        tangent_fit_y = ln_A[tangent_fit_window]

        # Perform a linear fit to find the tangent
        ln_A_tangent_coefficients = np.polyfit(tangent_fit_x, tangent_fit_y, 1)
        ln_A_tangent_slope, ln_A_tangent_intercept = ln_A_tangent_coefficients
        ln_A_tangent_line = ln_A_tangent_slope * h_v + ln_A_tangent_intercept

        # Calculate Urbach energy
        if ln_A_tangent_slope != 0:
            Eu = round((1 / ln_A_tangent_slope) * 1000, 2)  # Convert to meV
        else:
            Eu = None

        # Plot the tangent line for ln(A) at the maximum derivative
        ax2.plot(h_v, ln_A_tangent_line, '--', color="#FCCDC9", linewidth=2)  # Pink dashed line

        # Combine legends
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles1[:1] + handles2[:1], ["(A$\mathbf{h\\nu}$)²", "ln(A)"],
            frameon=False, fontsize=24, loc='upper left',
            prop={'weight': 'bold', 'size': 24, 'family': 'Arial'}
        )

        # Add a separate legend for Eg and Eu
        ax.text(
            0.045, 0.78, f"Eg = {Eg:.2f} eV\nEu = {Eu:.2f} meV",
            transform=ax.transAxes, fontsize=24, fontname="Arial", weight="bold",
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )

        # Save the plot
        output_image_path = os.path.join(uv_vis_fig_dir, f"Nr {sample_number} UV-Vis.png")
        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()

        # Append the calculated results to the results list
        results.append({
            "Sample Nr": sample_number,
            "UV-Vis calculated Eg (eV)": Eg,
            "UV-Vis calculated Eu (meV)": Eu
        })


    except Exception as e:
        print(f"Error processing sample number {sample_number}: {e}")
        continue


# Update the DataFrame with the calculated Eg and Eu values for all samples
for result in results:
    sample_number = result["Sample Nr"]
    uv_vis_eg_df.loc[uv_vis_eg_df["Sample Nr"] == sample_number, "UV-Vis calculated Eg (eV)"] = result["UV-Vis calculated Eg (eV)"]
    uv_vis_eg_df.loc[uv_vis_eg_df["Sample Nr"] == sample_number, "UV-Vis calculated Eu (meV)"] = result["UV-Vis calculated Eu (meV)"]

# Save the updated DataFrame to the output CSV file
uv_vis_eg_df.to_csv(output_csv_path, index=False)

# Print confirmation that the process has completed successfully
print("All plots and the updated CSV file have been successfully generated!")
