# Musical_Impact_on_Mental_Health_Senior_Thesis

#Overview
## The code first conducts proper cleaning techniques on the given dataset, Music & Mental Health Survey Results, including dropping unnecessary columns such as Foreign Language and age, adding an observation count column, and remapping the Music_effects column in order to be properly utilized in the below correltional analysis methods. After cleaning is completed, the two analysis methods described below are deployed. 

# Correlational Methods
## Ordinal Logistic Regression
### Ordinal Logit Regression is a statistical analysis method that explores the relationship between ordinal response variables and one or more explanatory variables. The ordinal variable, in this case the variable responsible for tracking the improved/worsened feeling of mental health after music, is a categorical variable with clear category levels. The explanatory variables are the variables that might influence or cause the ordinal one, which in this case is the various variables recorded during music interaction as well as the reported mental health symptoms. This regressional analysis will help with determining correlation between the different explanatory variables that are affecting the interaction between music and mental health and the categorical variables that measure that interaction.
## Spearman's Rank Correlation
### Spearman's Rank Correlation provides another way to use statistical analysis to explore the relationship between the mental health effect and different music listening frequencies and habits. Spearman's method uses the rank order that is within the ordinal data to fit the data more naturally since there isn't a uniform distance between the ranks. The data is also not linear, so this method doesn't have to rely on that in order to produce significant results. We can use the same cleaned columns as the ordinal logit regression since they both run similar testing and require similar formatting, including how both use the mental health effect column as a testing set.

# Running the Code
## This code can be ran in a simply python IDE. All necessary imports have been added, but ensure that the same libraries have been properly installed before attempting to run. Necessary libraries include Pandas, statsmodels.api, and scipy.stats. 

