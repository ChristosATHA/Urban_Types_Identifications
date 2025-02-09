# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the clustered data
data = pd.read_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/ACTIVIΤΙΕΣ_WITHOUT_F_DIV/clusters_Activities_Without_F_Div_15clusters.xlsx")

# Columns representing analysis fields (adjust this list to match the columns in your data)
analysis_fields = [
    'POP_DENS_ha_norm', 'F_DENS_ha_norm', 'POS_Influence_ha_norm',
]

# Determine the number of unique clusters for filename suffix
num_clusters = data['cluster'].nunique()
suffix = f"{num_clusters}clusters"  # e.g., "5clusters"

# Define the output folder for saving images and data
output_folder = r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/ACTIVIΤΙΕΣ_WITHOUT_F_DIV"

# Step 2: Create the Conventional Boxplot by Cluster Using Analysis Fields
# Calculate summary statistics for the boxplot: min, max, median, quartiles
min_values = data.groupby('cluster')[analysis_fields].min().reset_index()
q25_values = data.groupby('cluster')[analysis_fields].quantile(0.25).reset_index()
median_values = data.groupby('cluster')[analysis_fields].median().reset_index()
q75_values = data.groupby('cluster')[analysis_fields].quantile(0.75).reset_index()
max_values = data.groupby('cluster')[analysis_fields].max().reset_index()

# Merge the summary statistics into a single DataFrame
boxplot_stats = pd.concat([min_values, q25_values, median_values, q75_values, max_values], keys=['min', '25%', '50%', '75%', 'max'])
boxplot_stats = boxplot_stats.reset_index(level=0).rename(columns={'level_0': 'Statistic'})

# Save boxplot summary statistics to Excel with cluster count in filename
boxplot_stats.to_excel(fr"{output_folder}/boxplot_summary_statistics_Activities_Without_F_Div_{suffix}.xlsx", index=False)

# Melt the data for plotting purposes
melted_data = pd.melt(data, id_vars=['cluster'], value_vars=analysis_fields,
                      var_name='Analysis Field', value_name='Standardized Value')

# Automatically generate a color palette for each unique cluster
unique_clusters = sorted(data['cluster'].unique())  # Get the unique cluster IDs
palette = sns.color_palette("tab10", len(unique_clusters))  # Select a palette with enough colors
cluster_palette = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}  # Map colors to clusters

# Create a boxplot split by cluster
plt.figure(figsize=(18, 10))
sns.boxplot(x='Analysis Field', y='Standardized Value', hue='cluster', data=melted_data,
            showmeans=True, meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "10"},
            palette=cluster_palette)

# Add title and labels to the boxplot
plt.title('Boxplot of Analysis Fields by Cluster')
plt.xlabel('Analysis Field')
plt.ylabel('Standardized Values')
plt.xticks(rotation=90)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the boxplot as an image
plt.tight_layout()
boxplot_path = fr"{output_folder}/Step_E_Boxplot_Build_Density_Without_F_Div_{suffix}.png"
plt.savefig(boxplot_path)
plt.show()

# Step 3: Plot Mean Lines by Cluster Using Analysis Fields
# Group the data by cluster and calculate the mean for each analysis field
mean_values = data.groupby('cluster')[analysis_fields].mean()

# Save mean values data to Excel with cluster count in filename
mean_values.to_excel(fr"{output_folder}/mean_lines_data_Activities_Without_F_Div_{suffix}.xlsx")

# Plot mean lines for each cluster
plt.figure(figsize=(18, 10))
for cluster in mean_values.index:
    plt.plot(analysis_fields, mean_values.loc[cluster], marker='o', linestyle='--', color=cluster_palette[cluster], label=f'Cluster {cluster}')

# Add title and labels
plt.title('Mean Lines of Analysis Fields by Cluster')
plt.xlabel('Analysis Field')
plt.ylabel('Mean Standardized Value')
plt.xticks(rotation=90)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the mean lines plot as an image
plt.tight_layout()
mean_lines_path = fr"{output_folder}/Step_E_Diagram_Activities_Without_F_Div_{suffix}.png"
plt.savefig(mean_lines_path)
plt.show()

# Step 4: Create a Count Plot of Objects per Cluster
# Count the number of objects per cluster
cluster_counts = data['cluster'].value_counts().sort_index()

# Save count data to Excel with cluster count in filename
cluster_counts_df = cluster_counts.reset_index()
cluster_counts_df.columns = ['Cluster', 'Count']
cluster_counts_df.to_excel(fr"{output_folder}/count_plot_data_Activities_Without_F_Div_{suffix}.xlsx", index=False)

# Plot the counts per cluster
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette=[cluster_palette[i] for i in cluster_counts.index])

# Add title and labels to the bar plot
plt.title('Counts by Cluster ID')
plt.xlabel('Cluster ID')
plt.ylabel('Count')

# Annotate bars with counts
for index, value in enumerate(cluster_counts.values):
    plt.text(index, value + 50, str(value), ha='center')

# Save the count plot as an image
plt.tight_layout()
count_plot_path = fr"{output_folder}/Step_E_Histogram_Activities_Without_F_Div_{suffix}.png"
plt.savefig(count_plot_path)
plt.show()

# Step 5: Create a Summary Table for Means and Medians in Excel
# Group the data by cluster and calculate mean and median for each analysis field
# Columns representing raw analysis fields (adjust this list to match the columns in your data)
raw_analysis_fields = [
    'POP_DENS_ha', 'F_DENS_ha', 'POS_Influence_ha',
]
summary_stats = data.groupby('cluster')[analysis_fields].agg(['mean', 'median'])
summary_stats_raw = data.groupby('cluster')[raw_analysis_fields].agg(['mean', 'median'])

# Display the summary statistics in the console
print("Summary Statistics (Mean and Median per Cluster):")
print(summary_stats)
print("Summary Statistics (Mean and Median per Cluster):")
print(summary_stats_raw)

# Save the summary statistics to an Excel file with cluster count in filename
with pd.ExcelWriter(fr"{output_folder}/summary_statistics_Activities_Without_F_Div_{suffix}.xlsx") as writer:
    summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
    summary_stats_raw.to_excel(writer, sheet_name='Sum_Stats_raw')
    
print(f"Analysis complete. Saved files:\n"
      f"- Excel: boxplot_summary_statistics, mean_lines_data, count_plot_data, summary_statistics\n"
      f"- Images: {boxplot_path}, {mean_lines_path}, {count_plot_path}")