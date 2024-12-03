import pandas as pd
import numpy as np


data = "/scratch/1561626/LLM_data.csv"

df= pd.read_csv(data, encoding = 'ISO-8859-1', delimiter=',')

df['log_area']=np.log(df['area_km2'])
df['log_sr']=np.log(df['species_richness'])

skew_area = df['area_km2'].skew()
skew_log_area = df['log_area'].skew()
skew_sr = df['species_richness'].skew()
skew_log_sr = df['log_sr'].skew()

print('skew_area:', skew_area)
print('skew_log_area:', skew_log_area)
print('skew_sr:', skew_sr)
print('skew_log_sr:', skew_log_sr)