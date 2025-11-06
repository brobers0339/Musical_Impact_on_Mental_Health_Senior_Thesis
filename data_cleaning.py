'''
Read in the music and mental health dataset to do some cleaning in order to do correlational testing.
This includes dropping unwanted columns, adding an observation count column, remapping the effects to
maintain consistency and then remapping them to a numerical value instead. We also convert the likert
scale in the dataset to numerical values with the above defined method to define the explanatory columns
before dropping unwanted columns from that dataframe as well.

REDO DOCSTRING!
'''

import pandas as pd

def cleaned_data():
    categories = {
        'Improved' : 1,
        'No Effect' : 2,
        'Worsened' : 3,
        'Unknown' : 0
    }
    
    df = pd.read_csv("https://raw.githubusercontent.com/brobers0339/Musical_Impact_on_Mental_Health_Senior_Thesis/refs/heads/main/Music%26MentalHealthDataset.csv")

    df_cleaned = df.drop(['Timestamp', 
                          'Age', 
                          'Primary streaming service', 
                          'Exploratory', 
                          'Permissions', 
                          'BPM', 
                          'Foreign languages'], axis=1)  
    
    df_cleaned['obs_count'] = range(1, len(df_cleaned) + 1)

    df_cleaned['Effects_remapped'] = \
        df_cleaned['Music effects'].map({
                    'Improve' : 'Improved',
                    'No effect' : 'No Effect',
                    'Worsen' : 'Worsened',
        })



    df_cleaned['Effects_remapped'] = df_cleaned['Effects_remapped'].fillna('Unknown')
    df_cleaned['Effects_remapped_cat'] = df_cleaned['Effects_remapped'].map(categories)
    return df_cleaned