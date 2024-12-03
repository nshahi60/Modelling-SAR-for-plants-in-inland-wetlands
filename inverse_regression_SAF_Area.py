import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import os

# Define a function to read and process CSV for each country
def read_country_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',')
    
    # Replace 1.00E+31 with NaN
    df.replace(1.00E+31, np.nan, inplace=True)
    
    # Select the necessary columns
    selected_columns = ['Species richness', 'Area (km2)','SatAreaFrac']
    df = df[selected_columns]
    
    # Fill NaN values with the column mean (except for 'Species richness')
    df = df.apply(lambda col: col.fillna(col.mean()) if col.name != 'Species richness' else col)
    
    return df

# Function to perform the inverse relation plot for a single country
def plot_inverse_relation(ax, df, x_column, y_column, country_name):
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
    
    # Plot scatter and regression line
    ax.scatter(df[x_column], df[y_column], marker='o', alpha=0.7, color='blue')
    ax.plot(xx, yy_pred, linestyle='dashed', linewidth=4, color='red')
    
    # Annotate the plot with the metrics
    ax.text(0.08, 0.9, f'R²= {r_squared:.3f}', transform=ax.transAxes, fontsize=21)
    ax.text(0.08, 0.85, f'p-value= {p_value:.4f}', transform=ax.transAxes, fontsize=21)
    equation = f'y = {intercept:.2f} + {coefficient:.2f} * 1/x'
    ax.text(0.08, 0.8, f'{equation}', transform=ax.transAxes, fontsize=21)
    #ax.text(0.05, 0.75, f'Correlation= {correlation:.2f}', transform=ax.transAxes, fontsize=21)
    
    axes[i].legend([legend_text], loc='upper right', fontsize=22, frameon=False)
    #axes[i].legend([equation], loc='upper right', fontsize=17, frameon=False)
    axes[i].annotate(equation, xy=(0.54, 0.85), xycoords='axes fraction', ha='left', va='center', fontsize=22)
    
    print(f'{country_name}: {coefficient:.4f}  {p_value:.3f}  {r_squared:.3f}')
    
    # Set labels and title
    ax.set_xlabel(x_column, fontsize=19)
    ax.set_ylabel(y_column, fontsize=19)
    ax.set_title(f'{country_name}', fontweight="bold", fontsize=21)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

# Main code to process multiple country CSVs
if __name__ == "__main__":
    # List of CSV files for different countries
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
    # Define the variable to plot against 'Species richness'
    #variable_to_plot = 'Area (km2)'  # or 'SatAreaFrac'
    
    # Determine the number of subplots required (1 row per country)
    n_countries = len(csv_files)
    n_cols = 4  # Number of columns per row (adjustable)
    n_rows = (n_countries + n_cols - 1) // n_cols  # Calculate number of rows
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(36, 8 * n_rows))  # Adjust the size based on the number of rows/columns
    axes = axes.flatten()  # Flatten the axes array for easier iteration
    
    # Loop over CSV files and plot for each country
    for i, file_path in enumerate(csv_files):
        country_name = os.path.basename(file_path).split('_SAF')[0]  # Extract the country name from the file path
        df = read_country_data(file_path)
        
        # Plot the inverse relation for the selected variable
        plot_inverse_relation(axes[i], df, x_column, y_column, country_name)
    
    # Remove any empty subplots (if there are fewer countries than grid spaces)
    for j in range(i+1, n_rows * n_cols):
      fig.delaxes(axes[j])
      
    # Adjust layout and show the final figure
    plt.tight_layout()
    #plt.show()
    
    # Save the plot as a single image
    plt.savefig("/home/1561626/plots/inverse_relation_SAF_Area.png")
