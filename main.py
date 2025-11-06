def main():
    from data_cleaning import cleaned_data
    from convert_likert_scale import convert_likert_to_numeric
    from Spearman_Rank_Correlation import Spearman_Rank_Correlation, create_spearman_visualizations
    from Ordinal_Logistical_Regression import Ordinal_Logistical_Regression, create_Ordinal_Logistical_Regression_visualizations
    
    overall_df = cleaned_data()
    df_explanatory_cols = convert_likert_to_numeric(overall_df)

    olr_model = Ordinal_Logistical_Regression(overall_df, df_explanatory_cols)
    spearman_model = Spearman_Rank_Correlation(overall_df, df_explanatory_cols)
    
    create_Ordinal_Logistical_Regression_visualizations(olr_model)
    create_spearman_visualizations(spearman_model)
    
if __name__ == "__main__":
    main()