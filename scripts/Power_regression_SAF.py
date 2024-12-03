import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import os

def plot_power_relation(ax, df, x_column, y_column, country_name):
    # Drop rows with missing or infinite values in x_column or y_column
    df = df[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()

    # Log-transform x and y
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

    # Calculate regression metrics
    correlation = model.score(log_x, log_y)
    p_value = model_stats.pvalues[1]   # P-value of the slope
    r_squared = correlation
    coefficient = model.coef_[0][0]
    intercept = model.intercept_[0]

    # Plot data and regression line
    ax.scatter(log_x, log_y, s=12, color='blue', label='_nolegend_')
    ax.plot(log_x, model.predict(log_x), color='red', linestyle='dashed', linewidth=4, label='_nolegend_')

    # Annotate the plot with regression equation, R², correlation, and p-value
    equation = f'y = {coefficient:.2f} * x + {intercept:.2f}'
    metrics_text = f'R² = {r_squared:.3f}\nP-value = {p_value:.3f}'

    ax.set_title(f'{country_name}', fontweight="bold", fontsize=21)
    ax.set_xlabel('log(Area)',fontsize=19)
    ax.set_ylabel(f'log({y_column})',fontsize=19)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    print(f'{country_name}: {coefficient:.4f}  {p_value:.3f}  {r_squared:.3f}')

    # Annotate equation and metrics on the plot with adjusted text positions, for right bottom
    ax.annotate(equation, xy=(0.55, 0.25), xycoords='axes fraction', ha='left', va='center', fontsize=19)
    ax.annotate(metrics_text, xy=(0.55, 0.17), xycoords='axes fraction', ha='left', va='center', fontsize=19)
    #Top left
    #ax.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='center', fontsize=19)
    #ax.annotate(metrics_text, xy=(0.05, 0.87), xycoords='axes fraction', ha='left', va='center', fontsize=19)

if __name__ == "__main__":
    # List of CSV files for multiple countries
#    csv_files = [
#       "/home/1561626/GwHead_output/Australia_SAF_GwHead.csv",
#       "/home/1561626/GwHead_output/Bolivia_SAF_GwHead.csv",
#       "/home/1561626/GwHead_output/Cyprus_SAF_GwHead.csv",
#       "/home/1561626/GwHead_output/France_SAF_GwHead.csv",
#       "/home/1561626/GwHead_output/Greece_SAF_GwHead.csv",
#       "/home/1561626/GwHead_output/Italy_SAF_GwHead.csv",
#       "/home/1561626/GwHead_output/SouthKorea_SAF_GwHead.csv",
#       "/home/1561626/GwHead_output/Spain_SAF_GwHead.csv",
#       "/home/1561626/GwHead_output/Switzerland_SAF_GwHead.csv",
#       "/home/1561626/GwHead_output/UK_SAF_GwHead.csv",
#       "/home/1561626/GwHead_output/USA_SAF_GwHead.csv"
#        # Add other country CSV files here
#    ]
    csv_files = ["/home/1561626/SAF_output/Australia_SAF.csv",
              "/home/1561626/SAF_output/Bolivia_SAF.csv",
              "/home/1561626/SAF_output/South_Korea_SAF.csv",
              "/home/1561626/SAF_output/Switzerland_SAF.csv",
              "/home/1561626/SAF_output/UK_SAF.csv",
              "/home/1561626/SAF_output/Cyprus_SAF.csv",
              "/home/1561626/SAF_output/France_SAF.csv",
              "/home/1561626/SAF_output/Greece_SAF.csv",
              "/home/1561626/SAF_output/Italy_SAF.csv",
              "/home/1561626/SAF_output/Spain_SAF.csv",
              #"/home/1561626/SAF_output/Turkey_Malta_SAF.csv",
              #"/home/1561626/SAF_output/USA_SAF.csv"
    ]

    # The column names to use for the power relation (x = independent, y = dependent)
    x_column = 'Area (km2)'         # Example independent variable
    y_column = 'SatAreaFrac'  # Example dependent variable

    # Determine the number of subplots required (1 row per country)
    num_countries = len(csv_files)
    n_cols = 4  # Number of columns per row (adjustable)
    n_rows = (num_countries + n_cols - 1) // n_cols  # Calculate number of rows

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(36, 8 * n_rows))  # Adjust the size based on the number of rows/columns
    axes = axes.flatten()  # Flatten the axes array for easier iteration
    
#    # Set up subplots grid to accommodate multiple countries
#    num_countries = len(csv_files)
#    fig, axes = plt.subplots(1, num_countries, figsize=(20, 5))  # 1 row, as many columns as countries

    # If there's only one country, axes will not be an array, so we handle that case
    if num_countries == 1:
        axes = [axes]  # Convert to list for consistent handling

    # Loop over each country file
    for i, file_path in enumerate(csv_files):
        country_name = os.path.basename(file_path).split('_SAF')[0]  # Extract country name from file name
        
        # Read the CSV file for each country
        df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',')

        # Replace 1.00E+31 with NaN (specific to your data)
        df.replace(1.00E+31, np.nan, inplace=True)

        # Plot power relation for the specified x_column and y_column for each country
        plot_power_relation(axes[i], df, x_column, y_column, country_name)
        
    # Remove any empty subplots (if there are fewer countries than grid spaces)
    for j in range(i+1, n_rows * n_cols):
      fig.delaxes(axes[j])

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('/home/1561626/plots/power_relation_SAF_Area.png')

    # Show the plot
    #plt.show()
