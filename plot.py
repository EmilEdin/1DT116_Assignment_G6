import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
# Since your CSV doesn't have a header row, we name the columns manually
columns = ['Frame', 'Fade', 'Inc', 'Scale', 'Blur', 'GPUTotal', 'CPUTotal', 'TickTotal']
df = pd.read_csv('evaluation_data.csv', header=None, names=columns)

# Drop the first 5 frames to ignore the CUDA "warm-up" initialization time
df = df.iloc[5:] 

# 2. Set up the figure side-by-side (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- CHART 1: Kernel Average Bar Chart ---
# Calculate the average time for each of the four math kernels
kernel_means = df[['Fade', 'Inc', 'Scale', 'Blur']].mean()

# Plot the bar chart
ax1.bar(kernel_means.index, kernel_means.values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
ax1.set_title('Average Execution Time per Kernel')
ax1.set_ylabel('Time (ms)')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# --- CHART 2: Total Tick Box Plot ---
# Note: If you have your Sequential data in another CSV, you can load it here 
# and do: ax2.boxplot([seq_df['TickTotal'], df['TickTotal']], labels=['Sequential', 'CUDA'])
ax2.boxplot([df['TickTotal']], labels=['CUDA Version'])
ax2.set_title('Distribution of Total Tick Times (scenario.xml)')
ax2.set_ylabel('Total Time (ms)')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 3. Clean up the layout and display/save
plt.tight_layout()
plt.savefig('evaluation_charts.png', dpi=300) # Saves a high-res image for your report
plt.show()