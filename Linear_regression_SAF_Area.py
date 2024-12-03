'''
# Un comment the code below if you want to plot all parameters against each other at once.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import linregress

# Read CSV into DataFrame
df = pd.read_csv("/scratch/6766757/data/Australia/Wetland_species_AUS.csv", encoding='ISO-8859-1', delimiter=',')

# Replace 1.00E+31 with NaN
df.replace(1.00E+31, np.nan, inplace=True)

# Select relevant columns
selected_columns = ['Area', 'Species','Discharge','Groundwater depth','Evaporation', 'Groundwater recharge', 'Salinity', 'BOD','TP','NOXN']
df_selected = df[selected_columns]

# Impute missing values using the mean strategy
df_selected = df_selected.apply(lambda col: col.fillna(col.mean()) if col.name != 'Species' else col)

# Create subplots for each parameter
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Scatter Plots: Species vs Other Parameters', fontsize=16)

# Function to calculate R-squared, add trendline, and display equation
def plot_with_trendline(ax, x, y, xlabel, ylabel):
     sns.scatterplot(x=x, y=y, data=df_selected, ax=ax, color='blue')
     X = df_selected[x].values  # Convert to NumPy array
     y = df_selected[y].values  # Convert to NumPy array
     model = LinearRegression()
     model.fit(X.reshape(-1, 1), y)  # Reshape X to 2D array
     y_pred = model.predict(X.reshape(-1, 1))
     r2 = r2_score(y, y_pred)
     slope, intercept, r_value, p_value, std_err = linregress(X, y)
     equation = f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}\nR² = {r2:.2f}, p = {p_value:.4f}'
     ax.plot(X, y_pred, color='red', linewidth=2)
     ax.text(0.05, 0.9, equation, transform=ax.transAxes, fontsize=10, va='top', ha='left')
     ax.set_title(f'{ylabel} vs {xlabel}')
     ax.set_xlabel(xlabel)
     ax.set_ylabel(ylabel)
     ax.legend()

# Plot subplots
plot_with_trendline(axs[0, 0], 'Area', 'Species', 'Area', 'Species Richness')
plot_with_trendline(axs[0, 1], 'Groundwater recharge', 'Species', 'Groundwater Recharge', 'Species Richness')
plot_with_trendline(axs[1, 0], 'Discharge', 'Species', 'Discharge', 'Species Richness')
plot_with_trendline(axs[1, 1], 'Groundwater depth', 'Species', 'Groundwater depth', 'Species Richness')

plot_with_trendline(axs[0, 0], 'Evaporation', 'Species', 'Evaporation', 'Species Richness')
plot_with_trendline(axs[0, 1], 'Salinity', 'Species', 'Salinity', 'Species Richness')
plot_with_trendline(axs[1, 0], 'BOD', 'Species', 'BOD', 'Species Richness') 
plot_with_trendline(axs[1, 1], 'TP', 'Species', 'TP', 'Species Richness') 

plot_with_trendline(axs[1, 0], 'NOXN', 'Species', 'NOXN', 'Species Richness') 


# you can change the parameters here. Dont forget to change the number of rows and columns above. The order is row, column. eg. plot_with_trendline(axs[2, 0], 'TP', 'Species', 'TP (mg/L)', 'Species Richness')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save and display plot
plt.savefig('/home/1561626/SAF_output/scatter_plots_species_vs_parameters3.png')
plt.show()
'''

'''
# Use the code below if you want to plot only one parameter against species richness.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import linregress

# Read CSV into DataFrame
df = pd.read_csv("/scratch/1561626/test_data/Data wetlands(France (165)).csv", encoding='ISO-8859-1', delimiter=',')

# Replace 1.00E+31 with NaN
df.replace(1.00E+31, np.nan, inplace=True)

# Select all relevant columns
selected_columns_all = ['Area (km2)', 'Species richness','Species normalized']
#selected_columns_all = ['Area (km2)', 'Species richness','Species normalized','Discharge','Groundwater depth','Evaporation', 'Groundwater recharge', 'Salinity', 'BOD','bod','TP','ec','NOXN']
df_selected_all = df[selected_columns_all]

# Impute missing values using the mean strategy
df_selected_all = df_selected_all.apply(lambda col: col.fillna(col.mean()) if col.name != 'Species richness' else col)

# Create scatter plot for 'Species richness' and 'Area (km2)'
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Area (km2)', y='Species richness', data=df_selected_all)

# Linear Regression for 'Species' and 'Area (km2)'
X_all = df_selected_all[['Area (km2)']]
y_all = df_selected_all['Species richness']
model_all = LinearRegression()
model_all.fit(X_all, y_all)
y_pred_all = model_all.predict(X_all)

# Calculate R-squared
r2_all = r2_score(y_all, y_pred_all)

# Calculate p-value
slope, intercept, r_value, p_value, std_err = linregress(X_all.values.flatten(), y_all.values)

# Add trendline
plt.plot(X_all, y_pred_all, color='red', linewidth=2)

# Display equation of the slope, R-squared, and p-value in the legend
equation = f'y = {model_all.coef_[0]:.2f}x + {model_all.intercept_:.2f}'
legend_text = f'R²={r2_all:.2f}, pvalue={p_value:.4f}, {equation}'
plt.legend([legend_text], loc='upper right', fontsize=10, frameon=False) # frameon=False removes the box around the legend and loc='upper right' moves the legend to the upper right corner

# Display plot
plt.title('Scatter Plot: Species richness vs Area')
plt.xlabel('Area (sq.km)')
plt.ylabel('Species Richness')
plt.savefig('/scratch/1561626/test_data/scatter_plot_species_vs_area.png')
plt.show()
'''
# Use the code below to plot the SR-parameter relationship for different countries at once
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import linregress
import os

# List of CSV file paths for each country
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

# Determine the number of subplots required (1 row per country)
n_countries = len(csv_files)
n_cols = 4  # Number of columns per row (adjustable)
n_rows = (n_countries + n_cols - 1) // n_cols  # Calculate number of rows

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(36, 8 * n_rows))  # Adjust the size based on the number of rows/columns
axes = axes.flatten()  # Flatten the axes array for easier iteration

# Loop through each file and plot on the respective subplot
for i, file_path in enumerate(csv_files):
    # Extract country name from the file path (assuming the country is in the filename)
    country_name = os.path.basename(file_path).split('_SAF')[0].strip()

    # Read CSV into DataFrame
    df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',')

    # Replace 1.00E+31 with NaN
    df.replace(1.00E+31, np.nan, inplace=True)

    # Select all relevant columns
    selected_columns_all = ['SatAreaFrac', 'Area (km2)']
    df_selected_all = df[selected_columns_all]

    # Impute missing values using the mean strategy
    df_selected_all = df_selected_all.apply(lambda col: col.fillna(col.mean()) if col.name != 'Area (km2)' else col)
    
    # Drop rows where 'Species richness' is NaN
    df_selected_all = df_selected_all.dropna(subset=['Area (km2)'])

    # Scatter plot for 'Species richness' and 'Area (km2)'
    sns.scatterplot(x='Area (km2)', y='SatAreaFrac', data=df_selected_all, ax=axes[i])

    # Linear Regression for 'Species' and 'Area (km2)'
    X_all = df_selected_all[['Area (km2)']]
    y_all = df_selected_all['SatAreaFrac']
    model_all = LinearRegression()
    model_all.fit(X_all, y_all)
    y_pred_all = model_all.predict(X_all)

    # Calculate R-squared and p-value
    r2_all = r2_score(y_all, y_pred_all)
    slope, intercept, r_value, p_value, std_err = linregress(X_all.values.flatten(), y_all.values)

    # Add trendline
    axes[i].plot(X_all, y_pred_all, color='red', linewidth=4)

    # Add equation, R-squared, and p-value to the legend
    equation = f'y = {model_all.coef_[0]:.4f}x + {model_all.intercept_:.2f}'
    legend_text = f'R²={r2_all:.3f}, p-value={p_value:.3f}'
    axes[i].legend([legend_text], loc='upper right', fontsize=22, frameon=False)
    #axes[i].legend([equation], loc='upper right', fontsize=17, frameon=False)
    axes[i].annotate(equation, xy=(0.54, 0.85), xycoords='axes fraction', ha='left', va='center', fontsize=22)

    # Set title and labels
    axes[i].set_title(f'{country_name}', fontweight="bold", fontsize=21)
    axes[i].set_xlabel('Area', fontsize = 19)
    axes[i].set_ylabel('Saturated Area Fraction', fontsize=19)
    axes[i].tick_params(axis='x', labelsize=14)
    axes[i].tick_params(axis='y', labelsize=14)
    
    print(f'{country_name}: {model_all.coef_[0]:.4f}  {p_value:.3f}  {r2_all}')

# Remove any empty subplots (if there are fewer countries than grid spaces)
for j in range(i+1, n_rows * n_cols):
    fig.delaxes(axes[j])

# Adjust layout and save the figure
plt.tight_layout()
output_file = '/home/1561626/plots/Linear_regression_SAF_Area.png'
plt.savefig(output_file)
#plt.show()

