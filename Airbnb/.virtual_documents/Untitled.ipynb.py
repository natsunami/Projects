import os, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import xgboost as xgb 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, TruncatedSVD
from category_encoders.target_encoder import TargetEncoder
from category_encoders import LeaveOneOutEncoder





path = '../../Datasets'
file_name = 'listings.csv'
filepath = os.path.join(path,file_name)


df = pd.read_csv(filepath)


df.head()


df.columns


df.shape


isna_mask = (df.isna().sum()/df.shape[0]*100) < 30


isna_mask[isna_mask == False]


df = df.loc[:, isna_mask]


df_duplicated = df.duplicated(subset='id',keep=False)


df_duplicated[df_duplicated == True]


list(df.columns)
columns_to_keep = [
 'zipcode',
 'property_type',
 'room_type',
 'accommodates',
 'bathrooms',
 'bedrooms',
 'beds',
 'bed_type',
 'amenities',
 'price',
 'security_deposit',
 'cleaning_fee',
 'guests_included',
 'extra_people',
 'minimum_nights',
 'maximum_nights',
 'number_of_reviews'
 ]


df = df.loc[:,columns_to_keep]


df.shape


df_numerical = df.select_dtypes(include=['int','float'])
df_others = df.select_dtypes(exclude=['int','float'])


df_numerical


sns.heatmap(df_numerical.corr())


corr_mask = np.triu(np.ones_like(df_numerical.corr().abs(), dtype = bool))
df_numerical_corr_masked = df_numerical.corr().abs().mask(corr_mask)

numerical_col_to_remove = [ c for c in df_numerical_corr_masked.columns if any(df_numerical_corr_masked[c] > 0.8)]


df_numerical['mean_num_nights'] = (df_numerical['minimum_nights'] + df_numerical['maximum_nights'])/2
df_numerical.drop(['minimum_nights','maximum_nights'], axis=1, inplace=True)


priced_col = ['price','security_deposit','cleaning_fee','extra_people']

for col in priced_col:
    df_others[col] = df_others[col].str.replace('$','').str.replace(",","").astype('float')



def amenities_cleaning(x):
    x =  len(x.replace('{','').replace('}','').split(','))
    return x

df_others['amenities'] = df_others['amenities'].apply(lambda x: amenities_cleaning(x))


zipcode_mask = df_others['zipcode'].value_counts().index[df_others['zipcode'].value_counts() < 800]
mask_isin_zipcode = df_others['zipcode'].isin(zipcode_mask)
df_others['zipcode'][mask_isin_zipcode] = 'Other'
df_others['zipcode_clean'] = df_others['zipcode'].astype('str').apply(lambda x: x.replace(".0",""))
df_others.drop(['zipcode'], axis=1,inplace=True)


exotic_properties = df_others['property_type'].value_counts()[df_others['property_type'].value_counts() < 10].index
mask_isin_exotic = df_others['property_type'].isin(exotic_properties)
df_others['property_type'][mask_isin_exotic] = 'Exotic'


df_others['room_type'].unique()


df_others['bed_type'].unique()


df_others


df_numerical['mean_num_nights'] = (df_numerical['minimum_nights'] + df_numerical['maximum_nights'])/2


df.shape


df = pd.concat([df_others,df_numerical], axis=1)


df.shape


price_log = df['price'].apply(lambda x: np.log1p(x))


np.expm1(price_log)


fig, ax = plt.subplots()

sns.distplot(price_log, ax=ax)
plt.show()


df.columns
numerical_columns = ['amenities','security_deposit', 'cleaning_fee', 'extra_people','accommodates', 'bathrooms', 'bedrooms', 'beds', 'guests_included','number_of_reviews', 'mean_num_nights']
categorical_columns = ['property_type', 'room_type', 'bed_type','zipcode_clean']


y = df['price']
X = df.drop(['price'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X,price_log, shuffle=True, random_state= 42)


numerical_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),('scaler',RobustScaler())])

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('encoder',TargetEncoder())])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)])

#preprocessor_pca = Pipeline(steps=[('preprocess',preprocessor),('pca',T(n_components=0.90))])



X_train_processed = preprocessor.fit_transform(X_train,y_train)
X_test_processed = preprocessor.transform(X_test)


X_train_processed.shape


#Baseline model with linear regression:
lr = LinearRegression()

lr.fit(X_train_processed,y_train)
y_pred_train_lr = lr.predict(X_train_processed)
rmse_train_lr = mean_squared_error(y_train, y_pred_train_lr, squared=False)
r2_train_lr = r2_score(y_train, y_pred_train_lr)

#cv_rfr_lr_rmse = -cross_val_score(lr,X_train_processed, y_train, scoring=['neg_root_mean_squared_error','r2'], cv=10, n_jobs=-1).mean()
#cv_rfr_lr_r2 = -cross_val_score(lr,X_train_processed, y_train, scoring='r2', cv=10, n_jobs=-1).mean()
cross_validate(lr,X_train_processed, y_train, scoring=['neg_root_mean_squared_error','r2'], cv=10, n_jobs=-1)


#Baseline model with random forest regression:
rfr  = RandomForestRegressor()

rfr.fit(X_train_processed,y_train)
y_pred_train = rfr.predict(X_train_processed)
rmse_train_rfr = mean_squared_error(y_train, y_pred_train, squared=False)
r2_train_rfr = r2_score(y_train, y_pred_train)


cv_rfr  = cross_validate(rfr,X_train_processed, y_train, scoring=['neg_root_mean_squared_error','r2'], cv=5, n_jobs=-1)



cv_rfr


params = {'n_estimators':range(100,1000,50),
         'max_depth':range(1,30),
         'min_samples_leaf':range(1,20),
         'max_features':['auto','sqrt','log2']}

rand_search_rfr = RandomizedSearchCV(rfr,params,n_iter=20, cv=5, n_jobs=-1)
rand_search_rfr_results = rand_search_rfr.fit(X_train_processed, y_train)


pd.DataFrame(rand_search_rfr_results.cv_results_, columns=rand_search_rfr_results.cv_results_.keys()
            )


rand_search_rfr_results.best_score_


xboost = xgb.XGBRegressor()

xgb_cross_val = cross_validate(xboost, X_train_processed, y_train, scoring=['neg_root_mean_squared_error','r2'], cv=10,n_jobs=-1)


xgb_cross_val['test_r2'].mean()


params_xgb = {'n_estimators':range(100,1000,25),
         'max_depth':range(1,30),
         'min_samples_leaf':range(1,20),
         'learning_rate':np.linspace(0.001,0.3,num=50),
         'booster':['gbtree', 'gbdart'],
         'reg_alpha':np.linspace(0.01,0.5,num=50)}

rand_search_xgb = RandomizedSearchCV(xboost,params_xgb,n_iter=30, cv=5, n_jobs=-1)
rand_search_xgb_results = rand_search_xgb.fit(X_train_processed, y_train)


rand_search_xgb_results.best_params_



