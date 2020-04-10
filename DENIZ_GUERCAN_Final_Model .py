# timeit

# Student Name : Deniz Gürcan
# Cohort       : MSBA3 Castro

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports

# Importing relevant packages
import pydotplus
import random                  as rand                        # random number gen
import pandas                  as pd                          # data science essentials
import matplotlib.pyplot       as plt                         # data visualization
import seaborn                 as sns                         # enhanced data viz
import numpy                   as np                          # mathematical essentials
import pandas                  as pd                          # data science essentials
import matplotlib.pyplot       as plt                         # data visualization
import statsmodels.formula.api as smf                         # smf
from sklearn.model_selection   import train_test_split        # train-test split
from sklearn.linear_model      import LogisticRegression      # logistic regression
from sklearn.linear_model      import LinearRegression        # linear regression
from sklearn.metrics           import confusion_matrix        # confusion matrix
from sklearn.metrics           import roc_auc_score           # auc score
from sklearn.neighbors         import KNeighborsClassifier    # KNN for classification
from sklearn.neighbors         import KNeighborsRegressor     # KNN for regression
from sklearn.preprocessing     import StandardScaler          # standard scaler
from sklearn.tree              import DecisionTreeClassifier  # classification trees
from sklearn.tree              import export_graphviz         # exports graphics
from sklearn.externals.six     import StringIO                # saves objects in memory
from IPython.display           import Image                   # displays on frontend
from sklearn.model_selection   import GridSearchCV            # hyperparameter tuning
from sklearn.metrics           import make_scorer             # customizable scorer
from sklearn.ensemble          import RandomForestClassifier  # random forest
from sklearn.ensemble          import GradientBoostingClassifier # gbm
from sklearn.ensemble          import AdaBoostClassifier      # ADA Boost classifier 


################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')
# Determining and importing the dataset into Python

file        = 'Apprentice_Chef_Dataset.xlsx'
original_df = pd.read_excel(file)


################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# use this space for all of the feature engineering that is required for your
# final model

########################
# EMail Categories
########################
# First we split the data to make use of it
placeholder_lst = [] # defining an empty list 

# Creation of the for loop going over each value in the E-mail column
for index, col in original_df.iterrows(): 
    
    # Defining the split at the '@'
    email_spl = original_df.loc[index,'EMAIL'].split(sep = '@') 
    # Important to know that the split only works with txt objects
    
    # Filling the placeholder list with the slplit output
    placeholder_lst.append(email_spl)
    

# From list to dataframe 
email_spl_df = pd.DataFrame(placeholder_lst)


# Safety Precaution
original_df = pd.read_excel('Apprentice_Chef_Dataset.xlsx')

# Renaming the colum for concatenating
email_spl_df.columns = ['NAME' , 'EMAIL_DOMAIN']

# The Datasets have to be of equal length
original_df = pd.concat([original_df, email_spl_df.loc[:, 'EMAIL_DOMAIN']], # .loc = a safe way to do it
                   axis = 1)

# Professional domains splittet into their related industries 
tech_domain      = ['@apple.com', '@cisco.com', '@ibm.com', '@intel.com',
                    '@unitedtech.com', '@verizon.com',
                    '@microsoft.com']

fin_domain       = ['@amex.com', '@goldmansacs.com', '@jpmorgan.com', 
                    '@visa.com' ]

med_domain       = ['@jnj.com', '@unitedhealth.com',  '@pfizer.com', 
                    '@merck.com']

fnb_domain       = ['@mcdonalds.com', '@cocacola.com']

chem_domain      = ['@exxon.com', '@chevron.com',  '@dupont.com']

multi_domain     = ['@caterpillar.com', '@mmm.com', '@walmart.com', 
                    '@pg.com']
 
life_domain      = ['@disney.com', '@homedepot.com', '@nike.com',
                    '@travelers.com']

eng_domain       = ['@boeing.com', '@ge.org']

# Private and Spam/Junk domains
private_domain   = ['@gmail.com', '@yahoo.com', '@protonmail.com']

spam_domain      = ['@me.com', '@aol.com', '@hotmail.com',
                    '@live.com', '@msn.com', '@passport.com']

placeholder_lst = []  

# The loop groups observations by domain type
for domain in original_df['EMAIL_DOMAIN']:
       
        if '@' + domain in tech_domain: 
            placeholder_lst.append('Technology')
        
        elif '@' + domain in fin_domain:
            placeholder_lst.append('Finance') 
        
        elif '@' + domain in med_domain:
            placeholder_lst.append('Pharmaceutical')
        
        elif '@' + domain in fnb_domain:
            placeholder_lst.append('Food_n_Beverage')
        
        elif '@' + domain in multi_domain:
            placeholder_lst.append('Chemistry_Oil')
        
        elif '@' + domain in chem_domain:
            placeholder_lst.append('FMCG_Wholesale')
        
        elif '@' + domain in life_domain:
            placeholder_lst.append('Lifestyle')
        
        elif '@' + domain in eng_domain:
            placeholder_lst.append('Engineering')
        
        elif '@' + domain in private_domain:
            placeholder_lst.append('Private')
            
        elif '@' + domain in spam_domain:
            placeholder_lst.append('Junk')

        else:
            print('Not specified')
            
            
# Adding it to the original dataframe by concatonating
original_df['DOMAIN_KIND'] = pd.Series(placeholder_lst)

# One Hot encoding the newly created E-Mail Domains into to gain number forfor further analysis

# Creating binary dummies
# One Hot encoding
OH_domain = pd.get_dummies(original_df['DOMAIN_KIND'])

# Drop off columns with the categorial variables
original_df = original_df.drop('DOMAIN_KIND', axis = 1)
original_df = original_df.drop('EMAIL_DOMAIN', axis = 1)

# Combining the data
original_df = original_df.join(OH_domain)

# if your final model requires dataset standardization, do this here as well

# Preparation of a df based on our analysis; dropping categorial data
original_df_data = original_df.copy()
original_df_data = original_df_data.drop(['CROSS_SELL_SUCCESS', 'NAME', 'EMAIL', 
                                              'FIRST_NAME', 'FAMILY_NAME'], axis = 1)

# Defining the target variable
original_df_response = original_df.loc[ : , 'CROSS_SELL_SUCCESS'] 

#scaling into another df not original df

# Creating a StandardScaler() object
scaler = StandardScaler()

# Fit the scaler to the data
scaler.fit(original_df_data)

# Tranforming the data
X_scaled = scaler.transform(original_df_data)


# Convert the scaled data into a Dataframe
X_scaled_df = pd.DataFrame(X_scaled)


# View the results
X_scaled_df.describe().round(2)

# Re-add the column names to the scaled dataframe
X_scaled_df.columns = original_df_data.columns


################################################################################
# Train/Test Split
################################################################################

# use this space to set up testing and validation sets using train/test split

# Note: Be sure to set test_size = 0.25

# Initiaating a train data frame by splitting it accordingly
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
                                                    X_scaled_df,
                                                    original_df_response,
                                                    test_size = 0.25,
                                                    random_state = 222,
                                                    stratify = original_df_response)


# merging training data for statsmodels
scaled_df_train = pd.concat([X_train_scaled, y_train_scaled], axis = 1)




# Storing explanatory variables for each candidate model in a dictionary for convenience

candidate_dict = {

 # significant variables only
 'logit_sig'    : ["AVG_TIME_PER_SITE_VISIT",
                   "FOLLOWED_RECOMMENDATIONS_PCT",
                   "Finance",
                   "Technology"]

}

#train test slit with the new variables

X_scaled_df   =  X_scaled_df.loc[ : , candidate_dict['logit_sig']]

original_df_target =  original_df.loc[ : , 'CROSS_SELL_SUCCESS']

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
            X_scaled_df,
            original_df_target,
            test_size    = 0.25,
            random_state = 222,
            stratify     = original_df_target)




################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# use this space to instantiate, fit, and predict on your final model

# INSTANTIATING the model object with hyperparameters

# Defining the Grid Search
gsc = GridSearchCV(
    estimator = GradientBoostingClassifier(),
    param_grid={'n_estimators': (10, 50, 100, 500)},
    cv=5, scoring='roc_auc', verbose=0, n_jobs=-1)

grid_result = gsc.fit(X_scaled_df, original_df_target)
best_adb = grid_result.best_params_

# INSTANTIATING the model 
full_gbm_default = GradientBoostingClassifier(loss          = 'deviance',
                                              learning_rate = 0.02,
                                              n_estimators  = best_adb['n_estimators'],
                                              criterion     = 'friedman_mse',
                                              max_depth     = 1,
                                              warm_start    = False,
                                              random_state  = np.random)


# FIT step is needed as we are not using .best_estimator
full_gbm_default_fit = full_gbm_default.fit(X_train_scaled, y_train_scaled)


# PREDICTING based on the testing set
full_gbm_default_pred = full_gbm_default_fit.predict(X_test_scaled)



################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

test_score = roc_auc_score(y_true  = y_test_scaled,
                           y_score = full_gbm_default_pred).round(4)

print(test_score)
