import pandas as pd
import xarray as xr
import os

countries = ['Australia', 'Bolivia', 'Cyprus', 'France', 'Greece', 'Italy', 'South_Korea', 'Spain', 'Switzerland', 'UK', 'USA']
 
for country in countries:
    # File paths
    csv_file = f"/home/1561626/SAF_output/{country}_SAF.csv"
    nc_file = "/scratch/1561626/satAreaFrac_03_10_2024/estimateSatAreaFrac_monthAvg_1960-01-31_to_2019-12-31_avg.nc" # Replace with your NetCDF file path
    
    #csv_file = "/scratch/1561626/Wetland_data/USA/California_475.csv" # Replace with your CSV file path
     
    # Load the NetCDF file
    ds = xr.open_dataset(nc_file)
     
    # Load the CSV file containing latitude and longitude
    coords_df = pd.read_csv(csv_file, encoding='ISO-8859-1')
    
    # Dropping extra columns
    coords_df = coords_df.loc[:, ~coords_df.columns.str.contains('^Unnamed')]
     
    # Ensure your CSV has columns named 'lat' and 'lon' for latitude and longitude
    if 'lat' not in coords_df.columns or 'lon' not in coords_df.columns:
        raise ValueError("CSV must contain 'lat' and 'lon' columns.")
     
    # Function to extract data for each coordinate pair
    def extract_data(lat, lon):
        # Extract the data for the nearest lat/lon
        data_point = ds.sel(lat=lat, lon=lon, method="nearest")
        return data_point['satAreaFrac'].values.item()
     
    # Apply the extraction function to each row in the CSV and store the result in a new column
    coords_df['SatAreaFrac'] = coords_df.apply(lambda row: extract_data(row['lat'], row['lon']), axis=1)
     
    # Save the updated CSV file with the new column
    coords_df.to_csv(f"/home/1561626/SAF_output/{country}_SAF.csv", index=False)
 
print("Data extraction complete. Updated CSV saved.")