import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.svm import SVC, SVR
from statistics import mean, mode
from sklearn.impute import KNNImputer


df = pd.read_csv('Life_Expectancy_Data.csv')

''' Dummy variables '''

# df['Year'] = df['Year'].apply(str)
# cat_colmn = ['Year']
# # caterogy column = ['Year','Status','Country']
# for i in cat_colmn:
#     dummy = pd.get_dummies(df[i])
#     df.drop(i, axis=1, inplace=True)
#     df = pd.concat([df, dummy], axis=1)


''' correlation heatmap '''
# corrmat = df.corr()
# sns.heatmap(corrmat)

''' Data Analysis '''

df.drop(['Status', 'Country', 'Year'], axis=1, inplace=True)

''' Droping data of life expectancy== NaN '''
df.drop(df[df['Life expectancy '].isnull() == True].index, inplace=True)

df.drop('Hepatitis B', axis=1, inplace=True)  # having so many NaN values

df['Alcohol'].fillna(0.01, inplace=True)



# plt.scatter(df['Adult Mortality'], df['Life expectancy '])
# plt.show()


df['Total expenditure'].fillna(0, inplace=True)

''' bmi has corelation with last two colums but it has NaN values so we plot
histogram and fill mean value'''
# plt.hist(df[' BMI '])
# plt.show()
df[' BMI'].fillna(df[' BMI'].mean(), inplace=True)


df.drop(['GDP', 'Population'], axis=1, inplace=True)
df['Schooling'].fillna(df['Schooling'].mean(), inplace=True)
df['Income composition of resources'].fillna(
    df['Income composition of resources'].mean(), inplace=True)

df.reset_index(inplace=True)


def knn_impute(x, k):
    imp = KNNImputer(n_neighbors=k)
    return pd.DataFrame(imp.fit_transform(x))


i = knn_impute(df[['Polio', 'Diphtheria']], 10)
df['Polio'] = i.iloc[:, 0]
df['Diphtheria'] = i.iloc[:, 1]

i = knn_impute(df[[' thinness  1-19 years', ' thinness 5-9 years']], 10)
df[' thinness  1-19 years'] = i.iloc[:, 0]
df[' thinness 5-9 years'] = i.iloc[:, 1]


''' Removing Outliers '''

corrmat = df.corr()
top_corr_features = corrmat.index[abs(corrmat["Life expectancy "]) > 0.5]
plt.figure(figsize=(10, 10))
g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()


''' Model Preparation '''

x = df.drop(['Life expectancy ', 'infant deaths', 'Measles ',
             'under-five deaths ', 'Total expenditure', 'Polio'], axis=1)

y = df['Life expectancy ']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

''' Remove outliers only for x_train '''

df_ = pd.concat([x_train, y_train], axis=1)
df_.drop('index', axis=1, inplace=True)

df_.drop(df_[(df_['Life expectancy '] < 40) & (df_[' BMI'] > 40)].index, inplace=True)
df_.drop(df_[(df_['Life expectancy '] > 40) & (
    df_['Income composition of resources'] < 0.1)].index, inplace=True)

plt.scatter(df_['Schooling'], df_['Life expectancy '])
plt.show()
x_train = df_.drop('Life expectancy ', axis=1)
y_train = df_['Life expectancy ']

x_test.drop('index', axis=1, inplace=True)


def std(x_train, x_test):
    sc = StandardScaler()
    x_train = pd.DataFrame(sc.fit_transform(x_train))
    x_test = pd.DataFrame(sc.fit_transform(x_test))
    return x_train, x_test


x_train, x_test = std(x_train, x_test)

model = SVR()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
