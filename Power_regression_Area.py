import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
import statsmodels.api as sm
from scipy.stats import skew
import os

def plot_power_relation(ax, ax1, df, x_column, y_column, country_name, error_df):
    # Drop rows with missing or infinite values in x_column or y_column
    df = df[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()

    # Log-transform x and y
    df['log_x'] = np.log(df[x_column])
    df['log_y'] = np.log(df[y_column])
    log_x = np.log(df[x_column])
    log_y = np.log(df[y_column])


    # Drop rows with missing or infinite values in log_x or log_y
    log_x = log_x[~log_x.isin([np.inf, -np.inf, np.nan])]
    log_y = log_y[~log_y.isin([np.inf, -np.inf, np.nan])]

    # Ensure that log_x and log_y have the same number of elements after cleaning
    valid_idx = log_x.index.intersection(log_y.index)
    log_x = log_x.loc[valid_idx]
    log_y = log_y.loc[valid_idx]

    # Reshape data for Linear Regression
    log_x = log_x.values.reshape(-1, 1)
    log_y = log_y.values.reshape(-1, 1)

    # Fit Linear Regression model
    model = LinearRegression()
    model.fit(log_x, log_y)
    
    log_x_for_ols = log_x
    log_x_for_ols = sm.add_constant(log_x_for_ols)
    model_stats = sm.OLS(log_y, log_x_for_ols).fit()
    log_y_pred = model.predict(log_x)

    # Calculate regression metrics
    correlation = model.score(log_x, log_y)
    p_value = model_stats.pvalues[1]   # P-value of the slope
    r_squared = correlation
    coefficient = model.coef_[0][0]
    intercept = model.intercept_[0]
    error = mae(log_y, log_y_pred)
    residuals = log_y - log_y_pred
    skew_x = df['Area (km2)'].skew()
    skew_x_log = df['log_x'].skew()
    skew_y = df['Species richness'].skew()
    skew_y_log = df['log_y'].skew()
    
    error_row = {'Country':f'{country_name}',
                  'MAE':f'{error:.3f}',
                  'p-value':f'{p_value:.3f}',
                  'skew-x': f'{skew_x:.3f}',
                  'skew-x-log': f'{skew_x_log:.3f}',
                  'skew-y': f'{skew_y:.3f}',
                  'skew-y-log': f'{skew_y_log:.3f}'}
    error_df=pd.concat([error_df, pd.DataFrame([error_row])])
    
    #print(f'{country_name}: {error:.3f}   {p_value:.3f}')

    # Plot data and regression line
    ax.scatter(log_x, log_y, s=12, color='blue', label='_nolegend_')
    ax.plot(log_x, model.predict(log_x), color='red', linestyle='dashed', linewidth=4, label='_nolegend_')
    
    ax1.scatter(log_x, residuals)
    ax1.axhline(y=0, color='black', linewidth=3)
    ax1.set_title(f'{country_name}', fontweight="bold", fontsize=21)
    ax1.set_xlabel('log(Area)', fontsize=19)
    ax1.set_ylabel('Residuals', fontsize=19)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y',labelsize=14)

    # Annotate the plot with regression equation, R², correlation, and p-value
    equation = f'y = {coefficient:.2f} * x + {intercept:.2f}'
    metrics_text = f'R² = {r_squared:.3f}\np-value = {p_value:.3f}'

    ax.set_title(f'{country_name}', fontweight="bold", fontsize=21)
    ax.set_xlabel(f'log({x_column})',fontsize=19)
    ax.set_ylabel(f'log({y_column})',fontsize=19)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Annotate equation and metrics on the plot with adjusted text positions
    ax.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='center', fontsize=19)
    ax.annotate(metrics_text, xy=(0.05, 0.87), xycoords='axes fraction', ha='left', va='center', fontsize=19)
    
    print(f'{country_name}: {coefficient:.4f}  {p_value:.3f}  {r_squared:.3f}')
    
    return error_df

if __name__ == "__main__":
    # List of CSV files for multiple countries
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

    # The column names to use for the power relation (x = independent, y = dependent)
    x_column = 'Area (km2)'         # Example independent variable
    y_column = 'Species richness'  # Example dependent variable
    error_df = pd.DataFrame(columns=['Country', 'MAE', 'p-value', 'skew-x', 'skew-x-log', 'skew-y', 'skew-y-log'])

    # Determine the number of subplots required (1 row per country)
    num_countries = len(csv_files)
    n_cols = 4  # Number of columns per row (adjustable)
    n_rows = (num_countries + n_cols - 1) // n_cols  # Calculate number of rows

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(36, 8 * n_rows))  # Adjust the size based on the number of rows/columns
    axes = axes.flatten()  # Flatten the axes array for easier iteration
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize = (36,8 * n_rows)) #for residual plots
    axes1 = axes1.flatten()
    
#    # Set up subplots grid to accommodate multiple countries
#    num_countries = len(csv_files)
#    fig, axes = plt.subplots(1, num_countries, figsize=(20, 5))  # 1 row, as many columns as countries

    # If there's only one country, axes will not be an array, so we handle that case
    if num_countries == 1:
        axes = [axes]  # Convert to list for consistent handling
        axes1= [axes1]

    # Loop over each country file
    for i, file_path in enumerate(csv_files):
        country_name = os.path.basename(file_path).split('.')[0]  # Extract country name from file name
        
        # Read the CSV file for each country
        df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',')

        # Replace 1.00E+31 with NaN (specific to your data)
        df.replace(1.00E+31, np.nan, inplace=True)

        # Plot power relation for the specified x_column and y_column for each country
        error_df=plot_power_relation(axes[i], axes1[i], df, x_column, y_column, country_name, error_df)
        
    # Save the updated CSV file with the new column
    error_df.to_csv(f"/home/1561626/MAE_p-value/Power_Area_SR_error.csv", index=False)
        
    # Remove any empty subplots (if there are fewer countries than grid spaces)
    for j in range(i+1, n_rows * n_cols):
      fig.delaxes(axes[j])
      fig1.delaxes(axes1[j])

    # Adjust layout and save the plot
    fig.tight_layout()
    fig1.tight_layout()
    fig.savefig('/home/1561626/plots/power_relation_SR_Area.png')
    fig1.savefig('/home/1561626/plots/residual_power_SR_Area.png')

    # Show the plot
    #plt.show()
