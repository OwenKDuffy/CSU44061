import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

df = pandas.read_csv("tcd ml 2019-20 income prediction training (with labels).csv", index_col='Instance')
##print(df)
df['Year of Record'] = pandas.to_numeric(df['Year of Record'], errors='coerce').fillna(0)
df['Age'] = pandas.to_numeric(df['Age'], errors='coerce').fillna(0)
# df['Size of City'] = pandas.to_numeric(df['Size of City'], errors='coerce').fillna(0)
df['Body Height [cm]'] = pandas.to_numeric(df['Body Height [cm]'], errors='coerce').fillna(0)
gender_df = pandas.get_dummies(df['Gender'])

# jobs_df = pandas.get_dummies(df['Profession'])
# print(jobs_df)
# df_genders = df.Gender.values.astype(str)
# enc = OneHotEncoder(handle_unknown = 'ignore')
# X = ['male', 'female', 'other', 'female']
# g = enc.fit_transform(np.asarray(df_genders).reshape(-1, 1))
# g = enc.fit_transform(df_genders.reshape(-1, 1))
# print(g.toarray())
x_train = df[['Year of Record', 'Age', 'Body Height [cm]']].copy()
x_train['Male'] = gender_df['male'].copy()
x_train['Female'] = gender_df['female'].copy()
# x_train = pandas.merge(x_train, jobs_df, on = 'Instance')
# x_data = df[['Year of Record']].copy()
# x_data.to_csv("Sanitized.csv")
y_train = df['Income in EUR']
print(x_train)
#
tdf = pandas.read_csv("tcd ml 2019-20 income prediction test (without labels).csv", index_col='Instance')
##print(tdf)
tdf['Year of Record'] = pandas.to_numeric(tdf['Year of Record'], errors='coerce').fillna(0)
tdf['Age'] = pandas.to_numeric(tdf['Age'], errors='coerce').fillna(0)
# tdf['Size of City'] = pandas.to_numeric(tdf['Size of City'], errors='coerce').fillna(0)

tdf['Body Height [cm]'] = pandas.to_numeric(tdf['Body Height [cm]'], errors='coerce').fillna(0)
gender_tdf = pandas.get_dummies(tdf['Gender'])

# jobs_tdf = pandas.get_dummies(tdf['Profession'])

x_test = tdf[['Year of Record', 'Age', 'Body Height [cm]']].copy()
x_test['Male'] = gender_tdf['male'].copy()
x_test['Female'] = gender_tdf['female'].copy()
# x_test = pandas.merge(x_test, jobs_tdf, on = 'Instance')
# x_data = tdf[['Year of Record']].copy()
# x_data.to_csv("Sanitized.csv")
# y_test = tdf['Income']
print(x_test)
# #print(y_test)

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
results.to_csv("tcd ml 2019-20 income prediction submission with gender one hot.csv", header = "Instance, Income")
# The coefficients
# #print('Coefficients: \n', regr.coef_)
# # The mean squared error
# #print("Mean squared error: %.2f"
#       % mean_squared_error(Y_test, Y_pred))
# # Explained variance score: 1 is perfect prediction
# #print('Variance score: %.2f' % r2_score(Y_test, Y_pred))
#

# Plot outputs
# plt.scatter(X_test, Y_test,  color='black')
# plt.plot(X_test, Y_pred, color='blue', linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()
