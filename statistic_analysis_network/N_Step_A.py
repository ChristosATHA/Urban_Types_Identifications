import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the data from the Excel file
network_data = pd.read_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_imports/motorised_network_10km_radius_3035_segments_Disc_isl_Clipped_SPA_AN_NoZeros.xlsx")

# Step 2: Select the columns to normalize (columns 6 to 27)
# In Python, iloc[:, 5:26] means selecting columns 6 to 27 (zero-indexed)
columns_to_normalize = network_data.iloc[:, 5:27]

# Step 3: Standardize the data using Z-score normalization (only on the selected columns)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(columns_to_normalize)

# Step 4: Convert the normalized data back into a DataFrame and add "_norm" suffix to column names
normalized_columns = [col + "_norm" for col in columns_to_normalize.columns]
normalized_df = pd.DataFrame(data_normalized, columns=normalized_columns)

# Step 5: Combine the original data with the normalized columns
# This will ensure that the original dataset is preserved along with the normalized columns
final_df = pd.concat([network_data, normalized_df], axis=1)

# Step 6: Save the combined data (original + normalized) to an Excel file
output_excel_path = r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/normalized_network_data_combined_new.xlsx"
final_df.to_excel(output_excel_path, index=False)

# Print confirmation
print(f"Normalized data has been saved to {output_excel_path}")

# Step 7: Print the first 5 rows of the final combined data to check
print("First 5 rows of combined data (original + normalized):")
print(final_df.head())
