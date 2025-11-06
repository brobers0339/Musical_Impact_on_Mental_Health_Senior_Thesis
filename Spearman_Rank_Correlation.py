'''
Spearman's Rank Correlation
    Using Spearman's rank correlation testing, we measure the strength between the 
    different ranked categories in the dataset. After calculating the spearman's r-value
    and p-values, we write that information to a dataframe to store and eventually export
    to a csv file in order to be implemented into the final report and findings.
'''

#Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from scipy.stats import spearmanr

def Spearman_Rank_Correlation(df_cleaned, df_explanatory_cols):
    #Define the correlational matrix the values will be added to
    spearman_matrix = {'Category' : [], 'SpearmanR' : [], 'p-value' : []}

    for col in df_explanatory_cols:
        corr, pval = spearmanr(df_cleaned['Effects_remapped_cat'], df_explanatory_cols[col])
        spearman_matrix['Category'].append(col)
        spearman_matrix['SpearmanR'].append(corr)
        spearman_matrix['p-value'].append(pval)

    spearman_df = pd.DataFrame(spearman_matrix)

    spearman_df.to_csv('spearman_correlation.csv', index=False)
    return spearman_df
    
def create_spearman_visualizations(spearman_model):
    #Spearman's Correlation Visualizations
    df = spearman_model.sort_values('SpearmanR')
    
    plt.figure(figsize=(8,6))
    plt.hlines(y=df['Category'], xmin=0, xmax=df['SpearmanR'], color='lightgray')
    plt.scatter(df['SpearmanR'], df['Category'], c=df['SpearmanR'], cmap='coolwarm', s=80)
    plt.axvline(0, color='k', linewidth=0.8)
    plt.xlabel("Spearman's rho")
    plt.title("Lollipop: Spearman Correlations")
    plt.tight_layout()
    plt.savefig('spearman_correlation_lollipop.png')

    plt.figure(figsize=(8,6))
    sns.barplot(x='SpearmanR', y='Category', data=df, palette='vlag')
    plt.axvline(0, color='k', linewidth=0.8)
    plt.xlabel("Spearman's rho")
    plt.title("Spearman Correlations by Category")
    plt.tight_layout()
    plt.savefig('spearman_correlation_barplot.png')