'''
Ordinal Logistic Regression: 
    Using ordinal logistic regression, we attempt to predict the outcome of mental health 
    symptoms from each of the music categories to determine if there are any statistically
    significant correlations present between the two categories. 
    Fit the ordinal model using the Broyden-Fletcher-Goldfarb-Shanno
        algorithm, most popular minimizing optimization method as it improves the
        approximation of the computed Hessian matrix variables
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

def Ordinal_Logistical_Regression(df_cleaned, df_explanatory_cols):
    #Create ordinal logistical regression model using the cleaned dataset with remapped
    #effects column and the explanatory columns defined above.
    model = OrderedModel(
        df_cleaned['Effects_remapped_cat'],
        df_explanatory_cols,
        distr='logit'
    )

    res = model.fit(method='bfgs')

    #Fit model to a pandas dataframe to allow the data to be exported to a csv,
    #and then therefore exported to a table to be used in the final report
    model_df = pd.DataFrame({
        'Category' : res.model.exog_names,
        'Coef' : res.params,
        'Std Err' : res.bse,
        'z' : res.tvalues,
        'P>|z|' : res.pvalues,
        'CI_lower' : res.conf_int()[0],
        'CI_upper' : res.conf_int()[1]
    })

    #Write computed ordinal logistic regressional statistics to a csv file to be imported into final report
    model_df.to_csv('ordinal_logistic_regression.csv', index=True)
    return model_df

def create_Ordinal_Logistical_Regression_visualizations(olr_model):
    #Ordinal Logistical Regression Visualizations
    df = olr_model[(olr_model['Category'] != '0/1') & (olr_model['Category'] != '1/2') & (olr_model['Category'] != '2/3')]
    df= df.sort_values('Coef')
    plt.figure(figsize=(8,6))
    sns.barplot(x='Coef', y='Category', data=df, palette='vlag')
    plt.axvline(0, color='k', linewidth=0.8)
    plt.xlabel("Ordinal Logistic Regression Coefficient")
    plt.title("Ordinal Logistic Regression Correlations by Category")
    plt.tight_layout()
    
    plt.savefig('ordinal_logistic_regression_barplot.png')