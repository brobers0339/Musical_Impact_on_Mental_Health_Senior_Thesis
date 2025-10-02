import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/brobers0339/Musical_Impact_on_Mental_Health_Senior_Thesis/refs/heads/main/Music%26MentalHealthDataset.csv")

df_cleaned = df.drop(['Timestamp', 'Age', 'Primary streaming service', 'Exploratory', 'Permissions', 'BPM', 'Foreign languages'], axis=1)

df_cleaned['obs_count'] = range(1, len(df_cleaned) + 1)

def convert_likert_to_numeric(df, freq_cols, mapping=None):
    if mapping is None:
        mapping = {
            "Never" : 0,
            "Rarely" : 1,
            "Sometimes" : 2,
            "Often" : 3,
            "Always" : 4,

        }
    df_converted = df.copy()
    for col in freq_cols:
        df_converted[col] = df_converted[col].map(mapping).fillna(0)
    
    return df_converted

import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

#Ordinal Logit Regressional Model
df_cleaned['Effects_remapped'] = df_cleaned['Music effects'].map({
    'Improve' : 'Improved',
    'No effect' : 'No Effect',
    'Worsen' : 'Worsened',
    })
df_cleaned['Effects_remapped'] = df_cleaned['Effects_remapped'].fillna('Unknown')
df_cleaned['Effects_remapped_cat'] = df_cleaned['Effects_remapped'].map({
    'Improved' : 1,
    'No Effect' : 2,
    'Worsened' : 3,
    'Unknown' : 0
})

freq_cols = ['Frequency [Classical]', 
             'Frequency [Country]', 
             'Frequency [EDM]', 
             'Frequency [Folk]', 
             'Frequency [Gospel]', 
             'Frequency [Hip hop]', 
             'Frequency [Jazz]', 
             'Frequency [K pop]', 
             'Frequency [Latin]', 
             'Frequency [Lofi]', 
             'Frequency [Metal]', 
             'Frequency [Pop]', 
             'Frequency [R&B]', 
             'Frequency [Rap]', 
             'Frequency [Rock]', 
             'Frequency [Video game music]']

df_explanatory_cols = convert_likert_to_numeric(df_cleaned, freq_cols)
df_explanatory_cols = df_explanatory_cols.drop(['While working', 'Instrumentalist', 'Composer', 'Fav genre', 'Effects_remapped_cat', 'Music effects', 'Effects_remapped', 'obs_count'], axis=1)

model = OrderedModel(
    df_cleaned['Effects_remapped_cat'],
    df_explanatory_cols,
    distr='logit'
)

res = model.fit(method='bfgs')
print(res.summary())
