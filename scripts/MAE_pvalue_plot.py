import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

csv_files = ["/home/1561626/MAE_p-value/Power_Area_SR_error.csv",
              "/home/1561626/MAE_p-value/Linear_Area_SR_error.csv",
              "/home/1561626/MAE_p-value/Inverse_Area_SR_error.csv"]
              
combined_data = pd.DataFrame()

for i, file_path in enumerate(csv_files):

    # Read CSV into DataFrame
    df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',')
    
    selected_columns_all = ['MAE', 'p-value']
    df_selected_all = df[selected_columns_all]
    df_selected_all['p-value'] = df_selected_all['p-value'].replace(0, 0.001)
    df_selected_all['log_x'] = (np.log(df_selected_all['p-value']))
    #df_selected_all['log_x'] = df_selected_all['log_x'].replace([-np.inf], 0.000001)*(-1)
    df_selected_all['log_y'] = np.log(df_selected_all['MAE'])
    df_selected_all['label']=os.path.basename(file_path).split('_')[0].strip()
    
    combined_data = pd.concat([combined_data, df_selected_all])
    
# Creating the scatter plot
print(combined_data)
sns.scatterplot(data=combined_data, x='p-value', y='log_y', hue='label', style='label')

# Adding labels and title
plt.xlabel('log(p-value)')
plt.ylabel('log(Mean Absolute Error)')
plt.title('Comparison between different regression outputs')
plt.legend(title='Regression method')
plt.tight_layout()

# Display the plot
#plt.show()
output_file = '/home/1561626/plots/MAE_p_value_Area.png'
plt.savefig(output_file, dpi=300)


              