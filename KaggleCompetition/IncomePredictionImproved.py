import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
from sklearn import neural_network
df = pandas.read_csv("tcd ml 2019-20 income prediction training (with labels).csv", index_col='Instance')
trainingDataLength = len(df.index)
# print(trainingDataLength)
tdf = pandas.read_csv("tcd ml 2019-20 income prediction test (without labels).csv", index_col='Instance')
fulldf = pandas.concat([df, tdf], sort = True)
fulldf.to_csv("CombinedParams.csv")

print("Read data")
fulldf['Year of Record'] = pandas.to_numeric(fulldf['Year of Record'], errors='coerce').fillna(fulldf['Year of Record'].mean())
fulldf['Age'] = pandas.to_numeric(fulldf['Age'], errors='coerce').fillna(fulldf['Age'].mean())
fulldf['Size of City'] = pandas.to_numeric(fulldf['Size of City'], errors='coerce').fillna(fulldf['Size of City'].mean())
fulldf['Body Height [cm]'] = pandas.to_numeric(fulldf['Body Height [cm]'], errors='coerce').fillna(fulldf['Body Height [cm]'].mean())
print("Coerced Numeric data")

gender_df = pandas.get_dummies(fulldf['Gender'])
jobs_df = pandas.get_dummies(fulldf['Profession'])
Country_df = pandas.get_dummies(fulldf['Country'])
HairColor_df = pandas.get_dummies(fulldf['Hair Color'])
Degree_df = pandas.get_dummies(fulldf['University Degree'])
# Degree_df = Degree_df['Bachelor', 'Master', 'PhD'].copy()
# print("Created One Hot Encodings")
# valid_degrees = ['Bachelor', 'Master', 'PhD']
# enc = OrdinalEncoder(categories= valid_degrees, handle_unknown = 'ignore')
# Degree_df = enc.fit_transform(fulldf['University Degree'])
# print(Degree_df)

params = fulldf[['Year of Record', 'Age', 'Body Height [cm]', 'Size of City', 'Wears Glasses']].copy()
params['Male'] = gender_df['male'].copy()
params['Female'] = gender_df['female'].copy()
params['Bachelor'] = Degree_df['Bachelor'].copy()
params['Master'] = Degree_df['Master'].copy()
params['PhD'] = Degree_df['PhD'].copy()
params = pandas.merge(params, jobs_df, on = 'Instance')
params = pandas.merge(params, Country_df, on = 'Instance')
params = pandas.merge(params, HairColor_df, on = 'Instance')
# params = pandas.merge(params, Degree_df, on = 'Instance')

print("Merged into params")

# Normalizing is not providing enough of an improvement to justify the added run-time
# min_max_scaler = preprocessing.MinMaxScaler()
# scaled_values = min_max_scaler.fit_transform(params)
# params.loc[:,:] = scaled_values
# print("normalized")
stan_scaler = preprocessing.StandardScaler()
scaled_values = stan_scaler.fit_transform(params)
params.loc[:,:] = scaled_values
print("standardized")

x_train = params[:trainingDataLength]
x_test = params[trainingDataLength:]

# x_data.to_csv("Sanitized.csv")
y_train = df['Income in EUR']
y_train = y_train[:trainingDataLength]
# print(x_train)

# Create linear regression object
# regr = linear_model.MLPRegression()
regr = neural_network.MLPRegressor()

# Train the model using the training sets
regr.fit(x_train, y_train)
print("Trained models")
# Make predictions using the testing set
x_test['Income'] = regr.predict(x_test)
print("Made predictions")
# y_test = tdf['Income']
# results = pd.DataFrame()
results = x_test['Income'].copy()
# results.columns = ['Income']

print(results)
results.to_csv("tcd ml 2019-20 income prediction submission with all params one hot mean fillna MLP and Standardization.csv", header = "Instance, Income")
