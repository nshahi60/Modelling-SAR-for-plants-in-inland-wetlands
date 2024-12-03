import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
import statsmodels.api as sm
import os

# Define a function to read and process CSV for each country
def read_country_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',')
    
    # Replace 1.00E+31 with NaN
    #df.replace(1.00E+31, np.nan, inplace=True)
    
    # Select the necessary columns
    #selected_columns = ['Species richness', 'Area (km2)', 'SatAreaFrac', 'SatAreaFrac_20', 'SatAreaFrac_40']
    selected_columns = ['Species richness', 'Area (km2)']
    df = df[selected_columns]
    
    # Fill NaN values with the column mean (except for 'Species richness')
    df = df.apply(lambda col: col.fillna(col.mean()) if col.name != 'Species richness' else col)
    # Drop rows where 'Species richness' is NaN
    #df = df.dropna(subset=['Area (km2)'])

    df=df.dropna()

    return df

# Function to perform the inverse relation plot for a single country
def plot_inverse_relation(ax, ax1, df, x_column, y_column, country_name, error_df):
    model = LinearRegression()
    
    # Add a small constant (e.g., 1e-8) to the denominator to prevent division by zero
    model.fit(1 / (df[[x_column]] + 1e-8), df[y_column])
    
    # Calculate correlation
    correlation = np.corrcoef(df[y_column], model.predict(1 / (df[[x_column]] + 1e-8)))[0, 1]
    
    # Perform OLS regression
    X = sm.add_constant(1 / (df[[x_column]] + 1e-8))
    model_stats = sm.OLS(df[y_column], X).fit()
    
    r_squared = model_stats.rsquared
    p_value = model_stats.pvalues.iloc[1]
    coefficient = model.coef_[0]
    intercept = model.intercept_
    
    # Generate regression line
    xx = np.linspace(df[x_column].min(), df[x_column].max(), 1000)
    yy_pred = model.predict(1 / (xx.reshape(-1, 1) + 1e-8))
    #df['yy_pred'] = model.predict(1 / (xx.reshape(-1, 1) + 1e-8))
    
    error = mae(df[y_column], model.predict(1/(df[[x_column]]+1e-8)))
    print(f'{country_name}: {error:.3f}   {p_value:.3f}')
    error_row = {'Country':f'{country_name}',
                  'MAE':f'{error:.3f}',
                  'p-value':f'{p_value:.3f}'}
    error_df=pd.concat([error_df, pd.DataFrame([error_row])])
    
    # Plot scatter and regression line
    ax.scatter((df[x_column]), df['Species richness'], marker='o', alpha=0.7, color='blue')
    ax.plot(xx, yy_pred, linestyle='dashed', linewidth=4, color='red')
    
    # Annotate the plot with the metrics
    ax.text(0.08, 0.9, f'R²= {r_squared:.3f}', transform=ax.transAxes, fontsize=21)
    ax.text(0.08, 0.85, f'p-value= {p_value:.4f}', transform=ax.transAxes, fontsize=21)
    equation = f'y = {intercept:.2f} + {coefficient:.3f} * 1/x'
    ax.text(0.08, 0.8, f'{equation}', transform=ax.transAxes, fontsize=21)
    #ax.text(0.05, 0.75, f'Correlation= {correlation:.2f}', transform=ax.transAxes, fontsize=21)
    
    # Set labels and title
    ax.set_xlabel(f'Area (sq.km)', fontsize=19)
    ax.set_ylabel('Species richness', fontsize=19)
    ax.set_title(f'{country_name}', fontweight="bold", fontsize=21)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    #Residual plots
    residuals =df[y_column] - model.predict(1/(df[[x_column]]+1e-8))
    ax1.scatter(df[x_column], residuals)
    ax1.axhline(y=0, color='black', linewidth=3)
    ax1.set_title(f'{country_name}', fontweight="bold", fontsize=21)
    ax1.set_xlabel('Area (sq.km)', fontsize=19)
    ax1.set_ylabel('Residuals', fontsize=19)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y',labelsize=14)
    
    print(f'{country_name}: {coefficient:.4f}  {p_value:.3f}  {r_squared:.3f}')

    
    return error_df

#Main code to process multiple country CSVs
if __name__ == "__main__":
     #List of CSV files for different countries
    csv_files = [
    "/scratch/1561626/Wetland_data/Australia/Australia.csv",
    "/scratch/1561626/Wetland_data/Bolivia/Bolivia.csv",
    "/scratch/1561626/Wetland_data/South_Korea/South_Korea.csv",
    "/scratch/1561626/Wetland_data/Switzerland/Switzerland.csv",
    "/scratch/1561626/Wetland_data/UK/UK.csv",
    "/scratch/1561626/Wetland_data/Ottawa/Ottawa.csv",
    "/scratch/1561626/Wetland_data/Illinois/Illinois.csv",
    "/scratch/1561626/Wetland_data/China/China.csv",
    "/scratch/1561626/Wetland_data/Cyprus/Cyprus.csv",
    "/scratch/1561626/Wetland_data/France/France.csv",
    "/scratch/1561626/Wetland_data/Greece/Greece.csv",
    "/scratch/1561626/Wetland_data/Spain/Spain.csv",
    "/scratch/1561626/Wetland_data/Italy/Italy.csv",
    # Add more file paths as needed
    ]

#    csv_files = [
#        "/home/1561626/SAF_output/Australia_SAF.csv",
#        "/home/1561626/SAF_output/Bolivia_SAF.csv",
#        "/home/1561626/SAF_output/Cyprus_SAF.csv",
#        "/home/1561626/SAF_output/France_SAF.csv",
#        "/home/1561626/SAF_output/Greece_SAF.csv",
#        "/home/1561626/SAF_output/Italy_SAF.csv",
#        "/home/1561626/SAF_output/South_Korea_SAF.csv",
#        "/home/1561626/SAF_output/Spain_SAF.csv",
#        "/home/1561626/SAF_output/Switzerland_SAF.csv",
#        "/home/1561626/SAF_output/UK_SAF.csv",
#        "/home/1561626/SAF_output/USA_SAF.csv"
#        # Add other country CSV files here
#    ]
    
    # Define the variable to plot against 'Species richness'
    variable_to_plot = 'Area (km2)'  # or 'SatAreaFrac'
    error_df = pd.DataFrame(columns=['Country', 'MAE', 'p-value'])
    
    # Determine the number of subplots required (1 row per country)
    n_countries = len(csv_files)
    n_cols = 4  # Number of columns per row (adjustable)
    n_rows = (n_countries + n_cols - 1) // n_cols  # Calculate number of rows
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(36, 8 * n_rows))  # Adjust the size based on the number of rows/columns
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(36, 8 * n_rows))
    axes = axes.flatten()  # Flatten the axes array for easier iteration
    axes1 = axes1.flatten()
    
    # Loop over CSV files and plot for each country
    for i, file_path in enumerate(csv_files):
        country_name = os.path.basename(file_path).split('.')[0]  # Extract the country name from the file path
        df = read_country_data(file_path)
        
        # Plot the inverse relation for the selected variable
        error_df = plot_inverse_relation(axes[i], axes1[i], df, variable_to_plot, 'Species richness', country_name, error_df)
        
    # Save the updated CSV file with the new column
    error_df.to_csv(f"/home/1561626/MAE_p-value/Inverse_Area_SR_error.csv", index=False)
        
    # Remove any empty subplots (if there are fewer countries than grid spaces)
    for j in range(i+1, n_rows * n_cols):
      fig.delaxes(axes[j])
      fig1.delaxes(axes1[j])
      
    # Adjust layout and show the final figure
    fig.tight_layout()
    fig1.tight_layout()
    #plt.show()
    print(error_df)
    # Save the plot as a single image
    fig.savefig("/home/1561626/plots/inverse_relation_SR_Area.png")
    fig1.savefig("/home/1561626/plots/residual_inverse_SR_Area.png")
