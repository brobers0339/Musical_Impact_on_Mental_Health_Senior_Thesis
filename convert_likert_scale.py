'''
WRITE DOCSTRING!
'''
def convert_likert_to_numeric(df, mapping=None):
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
        
    df_converted = df_converted.drop(['While working', 'Instrumentalist', 'Composer', 'Fav genre', 'Music effects', 'Effects_remapped', 'obs_count', 'Effects_remapped_cat'], axis=1)

    return df_converted