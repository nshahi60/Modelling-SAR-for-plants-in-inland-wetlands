import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = "/scratch/1561626/LLM_data.csv"

df = pd.read_csv(data, encoding='ISO-8859-1', delimiter=',')

df['log(Area)'] = np.log(df['area_km2'])
df['log(Species richness)'] = np.log(df['species_richness'])

area_med = df.groupby('dataset')['area_km2'].median()
sr_med = df.groupby('dataset')['species_richness'].median()

#sns.boxplot(x='dataset', y=)
#plt.figure(figsize=(12,6))
#
#plt.subplot(1,2,1)
#sns.boxplot(x='dataset', y='log_area', data=df)
#
#plt.subplot(1,2,2)
#sns.boxplot(x='dataset', y='log_sr', data=df)
df_melted = df.melt(id_vars=['dataset'], 
                    value_vars=['log(Area)', 'log(Species richness)'], 
                    var_name='Metric', 
                    value_name='Log Value')

order_desired = ['AUS', 'UK', 'SKO', 'BOL', 'CYP', 'FRA', 'GRE', 'ITA', 'SPA', 'OTT', 'ILL', 'CHI', 'SWI']

# Create the grouped box plot
plt.figure(figsize=(12, 6))
plot=sns.boxplot(x='dataset', y='Log Value', hue='Metric', data=df_melted, order=order_desired, palette= 'tab10')
plot.set(xlabel="")
plt.xticks(ticks=plot.get_xticks(), labels=['Australia', 'U.K.', 'S.Korea', 'Bolivia', 'Cyprus', 'France', 'Greece', 'Italy', 'Spain', 'Ottawa', 'Illinois', 'China', 'Switzerland'], rotation=45, ha='right')
#plot.set_xticklabels()
plt.title('Distribution of log-transformed area and species richness data')
#plt.annotate(area_med, sr_med)
#plt.show()
plt.tight_layout()

plt.savefig('/home/1561626/plots/box_plot.png')







