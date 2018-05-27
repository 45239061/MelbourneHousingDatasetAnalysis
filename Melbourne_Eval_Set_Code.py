# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:02:34 2018

@author: Ol0f
"""


#dataset contains 19 features and 21797 rows

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid') #for data visualizations
from sklearn.model_selection import train_test_split
from scipy.stats import skew #for basic stats
import warnings #ignore warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn


train = pd.read_csv("Melbourne_Housing_train.csv")
test = pd.read_csv("Melbourne_Housing_Test.csv")
price = train['Price']

train.head()

train.dtypes

df = train.select_dtypes(include=['float64'])
df.head() 

print("Train set shape:", train.shape)
print("Test set shape:", test.shape)
#Train set shape: (21797, 19)
#Test set shape: (5450, 18) which is 20.0002% of training data


#Checking for missing values
missing = train.isnull().sum(axis=0).reset_index()
missing.columns = ['column_name', 'missing_count']
missing = missing.ix[missing['missing_count']>0]
missing = missing.sort_values(by='missing_count')
missing

"""column_name  missing_count
5        Distance              1
6        Postcode              1
13    CouncilArea              3
16     Regionname              3
17  Propertycount              3
14      Lattitude           5013
15     Longtitude           5013
7        Bedroom2           5162
8        Bathroom           5166
9             Car           5460
10       Landsize           7423
12      YearBuilt          12124
11   BuildingArea          13231"""

#since our target is price, lets do some inspection with it
plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train.Price.values))
plt.xlabel('Variation Of Price', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()
#here we just plotted the price for all the rows

#plotting a frequency dist and checking
plt.figure(figsize=(12, 10))
sns.distplot(train.Price.values, bins=50, kde=False)
plt.xlabel('Price', fontsize=12)
plt.show()

#Before doing anything else let's take care of missing values

#For filling the missing values in yearbuilt, we will assume that places nearby or close were
#built in the same year i.e. we will group up by suburb

median_year = train.groupby(['Suburb'])["YearBuilt"].median()

def fillna_year(row, median_year):
        year = median_year.loc[row["Suburb"]]
        return year

train["YearBuilt"] = train.apply(lambda row : fillna_year(row, median_year) if np.isnan(row['YearBuilt']) else row['YearBuilt'], axis=1)
#still 78 missing values are there, probably because they could'nt be grouped, we'll try filling it 
#with median(since 0.3% only)
most_common = train['YearBuilt'].value_counts().index[0]
train['YearBuilt'].fillna(most_common, inplace=True)

median_pc = train.groupby(['Suburb'])["Postcode"].median()

def fillna_postcode(row, median_pc):
        postcode = median_pc.loc[row["Suburb"]]
        return postcode

train["Postcode"] = train.apply(lambda row : fillna_year(row, median_pc) if np.isnan(row['Postcode']) else row['Postcode'], axis=1)

most_common = train['Postcode'].value_counts().index[0]
train['Postcode'].fillna(most_common, inplace=True)

most_common = train['Regionname'].value_counts().index[0]
train['Regionname'].fillna(most_common, inplace=True)

most_common = train['CouncilArea'].value_counts().index[0]
train['CouncilArea'].fillna(most_common, inplace=True)

median_lat = train.groupby(['Suburb', 'Postcode', 'CouncilArea'])["Lattitude"].median()

def fillna_lat(row, median_lat):
        lat = median_lat.loc[row["Suburb"], row['Postcode'], row["CouncilArea"]]
        return lat

train["Lattitude"] = train.apply(lambda row : fillna_lat(row, median_lat) if np.isnan(row['Lattitude']) else row['Lattitude'], axis=1)
#still 53 missing values left in latitude, which is only 0.24%, safe to fill it with median
most_common = train['Lattitude'].value_counts().index[0]
train['Lattitude'].fillna(most_common, inplace=True)

#we will apply the same technique for longitude
median_long = train.groupby(['Suburb', 'Postcode', 'CouncilArea'])["Longtitude"].median()

def fillna_long(row, median_long):
        long = median_long.loc[row["Suburb"], row['Postcode'], row["CouncilArea"]]
        return long

train["Longtitude"] = train.apply(lambda row : fillna_long(row, median_long) if np.isnan(row['Longtitude']) else row['Longtitude'], axis=1)
#still 53 missing values left in longtitude, which is only 0.24%, safe to fill it with median
most_common = train['Longtitude'].value_counts().index[0]
train['Longtitude'].fillna(most_common, inplace=True)


train['Propertycount'].fillna(train['Propertycount'].median(), inplace=True)

train['Distance'].fillna(train['Distance'].median(), inplace=True)

#Year Built contains a lot of missing values
plt.figure(figsize=(12, 10))
sns.boxplot(train.YearBuilt.values, train.Price.values)
plt.xlabel('YearBuilt', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.show()
#seems like some of the old houses have a high median value of price due to antique nature

#room and bedroom2 are actually the same column, replace bedroom2 accordingly
train.replace(to_replace='Bedroom2', value='Rooms', inplace=True)
train.drop('Bedroom2', axis=1, inplace=True)

most_common = train['Bathroom'].value_counts().index[0]
train['Bathroom'].fillna(most_common, inplace=True)

train['TotalRooms'] = train['Rooms'] + train['Bathroom']
train.drop('Rooms', axis=1, inplace=True)
train.drop('Bathroom', axis=1, inplace=True)

median_car = train.groupby(['TotalRooms'])["Car"].median()
def fillna_car(row, median_car):
        car = median_car.loc[row["TotalRooms"]]
        return car
train['Car'].fillna(train['Car'].median(), inplace=True)


#fill landsize NA and zeroes with median andd do the same for building area
#then if landsize < building area swap the two values(check row 87 port melbourne)
train['Landsize'].fillna(train['Landsize'].median(), inplace=True)
train['BuildingArea'].fillna(train['BuildingArea'].median(), inplace=True)
train['Landsize'] = train['Landsize'].replace(0, train['Landsize'].median())
train.Landsize, train.BuildingArea = np.where(train.Landsize < train.BuildingArea, [train.BuildingArea, train.Landsize], [train.Landsize, train.BuildingArea])

#Create a new lawnspace column
train['LawnSpace'] = train['Landsize'] - train['BuildingArea']
train.drop('Landsize', axis=1, inplace=True)
train.drop('BuildingArea', axis=1, inplace=True)
train.shape

#Checking relation of prices and suburbs
train.Suburb.value_counts()
median_price = train.groupby(['Suburb'])["Price"].median()
train.drop('Price', axis=1, inplace=True)

#LabelEncoding Categorical Variables
from sklearn.preprocessing import LabelEncoder
cols = ('Suburb', 'Method', 'SellerG', 'CouncilArea', 'Regionname', 'Type')
#Applying LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))

#Plotting heat map to see interactions between variables in our dataset
colormap = plt.cm.RdBu
fig = plt.figure(figsize=(200,200))
fig.add_subplot(1,1,1)
plt.title('Correlation of Features', y=1.05, size=8)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
#It can be seen that there is no strong interrelation between the variables which imply that we
#have less redundant data in our dataset which is good

#Plotting the relation between Price and Total Rooms
plt.figure(figsize=(12, 10))
sns.boxplot(train.TotalRooms.values, train.Price.values)
plt.xlabel('Total Rooms', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.title("Distribution of Price with "+ 'Total Rooms', fontsize=15)
plt.show()


#Checking skewness of all numerical values
numeric_feats = train.dtypes[train.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 0.75]
skewness.drop('Lattitude', inplace=True)
skewness.drop('Longtitude', inplace=True)
skewness.drop('CouncilArea', inplace=True)
#skewness.drop('Rooms', inplace=True)
skewness.drop('Suburb', inplace=True)
skewness.drop('SellerG', inplace=True)
skewness.drop('Regionname', inplace=True)
skewness.drop("YearBuilt", inplace=True)
#skewness.drop('Method_S', inplace=True)
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

#BoxCox transforming the skewed features
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    train[feat] = boxcox1p(train[feat], lam)

#Applying log(1+x) transformation to price
price = np.log1p(price)
print(train.shape)

#dropping features not important as per our model
train.drop('Suburb', axis=1, inplace=True)
train.drop('Regionname', axis=1, inplace=True)
train.drop('Car', axis=1, inplace=True)

#Splitting train test
X_train, X_test, Y_train, Y_test = train_test_split(train, price, test_size = 0.2, random_state = 7)


from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.055,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_rf = RandomForestRegressor(n_estimators=600, criterion="mse", random_state=7)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.098, n_estimators=3000,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

#Testing lgb model performance
model_lgb.fit(X_train, Y_train)
Y_pred1 = model_lgb.predict(X_test)

Y_pred1 =np.expm1(Y_pred1)
Y_pred1 = Y_pred1.astype(int)
Y_test = np.expm1(Y_test)

from sklearn.metrics import r2_score
r2_score(Y_test, Y_pred1) #r2_score:0.7731, rmsle:0.0903

#Testing RandomForest performance
model_rf.fit(X_train, Y_train)
Y_pred2 = model_rf.predict(X_test)

Y_pred2 =np.expm1(Y_pred2)
Y_pred2 = Y_pred2.astype(int)
r2_score(Y_test, Y_pred2) #r2_score:0.7674, rmsle:0.0914

#Testing GradientBoosting Performance
GBoost.fit(X_train, Y_train)
Y_pred3 = GBoost.predict(X_test)

Y_pred3 =np.expm1(Y_pred3)
Y_pred3 = Y_pred3.astype(int)
r2_score(Y_test, Y_pred3) #r2_score:0.7859, rmsle:0.0864

#Making the final prediction list by giving more weightage to GradientBoostingModel
Y_pred_final = Y_pred3*0.5 + (Y_pred2 + Y_pred1)*0.25
Y_pred_final = Y_pred_final.astype(int)
r2_score(Y_test, Y_pred_final) #final r2_score:0.79000, final rmsle:0.0856

#Calculating rmsle error for final model
import math
from math import sqrt

pred_list = []
test_list = []

for num in Y_pred_final:
    num = math.log10(num)
    pred_list.append(num)
    
for num in Y_test:
    num = math.log10(num)
    test_list.append(num)

rmsle = sqrt(mean_squared_error(test_list, pred_list))
print(rmsle) #rmsle error: 0.0856