import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

df = pandas.read_csv("tcd ml 2019-20 income prediction training (with labels).csv", index_col='Instance')
trainingDataLength = len(df.index)
# print(trainingDataLength)
tdf = pandas.read_csv("tcd ml 2019-20 income prediction test (without labels).csv", index_col='Instance')
fulldf = pandas.concat([df, tdf], sort = True)
fulldf.to_csv("CombinedParams.csv")

# print(fulldf)
fulldf['Year of Record'] = pandas.to_numeric(fulldf['Year of Record'], errors='coerce').fillna(0)
fulldf['Age'] = pandas.to_numeric(fulldf['Age'], errors='coerce').fillna(0)
# # df['Size of City'] = pandas.to_numeric(df['Size of City'], errors='coerce').fillna(0)
fulldf['Body Height [cm]'] = pandas.to_numeric(fulldf['Body Height [cm]'], errors='coerce').fillna(0)
gender_df = pandas.get_dummies(fulldf['Gender'])
#
jobs_df = pandas.get_dummies(fulldf['Profession'])
# # print(jobs_df)

params = fulldf[['Year of Record', 'Age', 'Body Height [cm]']].copy()
params['Male'] = gender_df['male'].copy()
params['Female'] = gender_df['female'].copy()
params = pandas.merge(params, jobs_df, on = 'Instance')

x_train = params[:trainingDataLength]
x_test = params[trainingDataLength:]

# x_data.to_csv("Sanitized.csv")
y_train = df['Income in EUR']
y_train = y_train[:trainingDataLength]
# print(x_train)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
x_test['Income'] = regr.predict(x_test)
# y_test = tdf['Income']
# results = pd.DataFrame()
results = x_test['Income'].copy()
# results.columns = ['Income']

print(results)
# results.to_csv("tcd ml 2019-20 income prediction submission with gender and jobs one hot.csv", header = "Instance, Income")
