import pandas as pd
import numpy as np
import os

countries = ['Cyprus', 'France', 'Greece', 'Spain']

result_list=[]

for country in countries:
    csv_inland = f"/scratch/1561626/Wetland_data/{country}/{country}.csv"
    csv_species = f"/scratch/1561626/Wetland_data/{country}/{country}_Flora.csv"
    csv_info = f"/scratch/1561626/Wetland_data/{country}/{country}_General_wetland_info.csv"
    
    df_inland = pd.read_csv(csv_inland, encoding='ISO-8859-1', delimiter=',')
    df_species = pd.read_csv(csv_species, encoding='ISO-8859-1', delimiter=',')
    df_info = pd.read_csv (csv_info, encoding='ISO-8859-1', delimiter=',')
    
    # Dropping extra columns
    df_inland = df_inland.loc[:, ~df_inland.columns.str.contains('^Unnamed')]
    
    selected_columns = ['ID', 'Natural / Artificial', 'Wetland type', 'Water salinity', 'Freshwater entry', 'Water presence', 'Biological significance (artificial wetlands)']
    df_selected_info = df_info[selected_columns]
    
    df_inland=df_inland.merge(df_selected_info, how='left', on='ID')
    
    #Extracting species data for each wetland
    selected_col_species = ['ID', 'Family', 'Species']
    df_species_selected = df_species[selected_col_species]
    df_families_selected = df_species[selected_col_species]
    
    # Convert the 'Species' column to string and replace NaNs with empty strings
    df_species_selected['Species'] = df_species_selected['Species'].fillna('').astype(str)
    
    df_species_selected=df_species_selected.groupby('ID')['Species'].apply(','.join).reset_index()
    
    df_inland=df_inland.merge(df_species_selected, how='left', on='ID')
    
    #Extracting maximum occurrence species
    df_species_list=df_inland[['ID','Species']]
    df_species_list['Species']=df_species_list['Species'].str.split(',')
    df_species_list=df_species_list.explode('Species')
    top_species=df_species_list['Species'].value_counts().head(5)
    print(top_species)
    
    #Extracting maximum occurrence family
    df_families_selected['Family'] = df_families_selected['Family'].fillna('').astype(str)
    df_families_selected=df_families_selected.groupby('ID')['Family'].apply(','.join).reset_index()
    df_inland=df_inland.merge(df_families_selected, how='left', on='ID')
    df_family_list=df_inland[['ID','Family']]
    df_family_list['Family']=df_family_list['Family'].str.split(',')
    df_family_list=df_family_list.explode('Family')
    top_family=df_family_list['Family'].value_counts().head(5)
    print(top_family)
    
    result_list.append(df_inland)
    
    output_file = f"/scratch/1561626/Wetland_data/{country}/{country}_expanded.csv"
    df_inland.to_csv(output_file, index=False)

#final_df = pd.concat(result_list, ignore_index=True)



