import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Input CSV file paths
uv_data_path = "/Users/mac/Desktop/WBG experiment/Python/PL Data analyze/241211/Remove noise csv/UV data.csv"

# Load UV data
df = pd.read_csv(uv_data_path)

# Create output folders
output_dir = os.path.join(os.path.dirname(uv_data_path), "Output")
uv_vis_urbach_dir = os.path.join(output_dir, "UV Vis Urbach Energy")
os.makedirs(uv_vis_urbach_dir, exist_ok=True)

# User input: Specify sample number
sample_number = 79  # Specify the sample number

# Extract wavelength and sample data
wavelength = df['Wavel.'].values.astype(float)  # Extract wavelength from the first column
absorbance = df[str(sample_number)].values.astype(float)  # Extract absorbance for the specified sample

# Convert wavelength to photon energy (eV)
h_v = 1240 / wavelength  # Photon energy
ln_absorbance = np.log(absorbance)  # Natural log of absorbance

# Dynamically calculate y-axis range
x_axis_range = (h_v >= 1.41) & (h_v <= 1.99)  # x-axis range
y_min = np.min(ln_absorbance[x_axis_range])
y_max = np.max(ln_absorbance[x_axis_range]) * 1.1  # Dynamically calculate y_max

# Plot the data
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the original curve
ax.plot(h_v, ln_absorbance, label="ln(A)", color='black', linewidth=2)

# Format the axes and labels
ax.set_xlim(1.41, 1.99)
ax.set_ylim(y_min, y_max)  # Automatically determined y_max

# Set axis labels with Arial font and bold weight
ax.set_xlabel("$\mathbf{h\\nu}$ (eV)", fontsize=32, weight='bold', labelpad=10, fontname="Arial")
ax.set_ylabel("ln(A)", fontsize=32, weight='bold', labelpad=10, fontname="Arial")

# Customize ticks and set bold Arial font for tick labels
ax.tick_params(axis='both', which='both', direction='in', width=2, length=6, labelsize=24, labelcolor='black')
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname("Arial")
    label.set_weight("bold")

# Set bold line widths for axes
for spine in ax.spines.values():
    spine.set_linewidth(2)

# Add legend with Arial font and bold weight
ax.legend(
    frameon=False, fontsize=24, loc='upper left',
    prop={'weight': 'bold', 'size': 24, 'family': 'Arial'}
)

# Save the plot to the updated folder
output_image_path = os.path.join(uv_vis_urbach_dir, f"Nr {sample_number} Urbach Energy.png")
plt.tight_layout()
plt.savefig(output_image_path)
plt.show()

print(f"The plot for sample {sample_number} has been successfully saved to {output_image_path}!")
