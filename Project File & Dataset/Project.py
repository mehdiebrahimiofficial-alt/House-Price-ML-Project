import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV 
from imblearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
import pickle

warnings.filterwarnings('ignore')


Data=pd.read_csv('D:\Documents\Python\VS code\Projects\House_Price_RLProject\House Price India.csv')

print('---Data information---')
print(Data.info())

print('---Dataset Columns---')
print(Data.columns)
print(Data.shape)
print(Data.isnull().sum())

print('---head of Data---')
print(Data.head())


print('---Data Statistic Summary---')
print(Data.describe())


Data['Date']=pd.to_datetime(Data['Date'],errors='coerce')

Data['Year']=Data['Date'].dt.year
Data['Month']=Data['Date'].dt.month


Data.drop(columns=['Date','id'],inplace=True)
print(list(Data.columns))


print('---feature engineering---')
Data.fillna(Data.median(numeric_only=True),inplace=True)

Data['Total Rooms']=Data['number of bedrooms']+Data['number of bathrooms']
Data['House Age']=2026-Data['Built Year']
Data['Renovation Age']=2026-Data['Renovation Year']
Data['Total Area']=Data['living area']+Data['lot area']

print(Data)


print('---visualization---')

plt.figure(figsize=(10,10))
sns.histplot(data=Data['Price'],bins=100,kde=True)
plt.title('Price visualization')
plt.savefig('Price visualization.png')
plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(Data.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title('Corr of features')
plt.savefig('Corr of features.png')
plt.show()


plt.figure(figsize=(10,10))
sns.scatterplot(x=Data['living area'],y=Data['Price'])
plt.title('Area---Price')
plt.savefig('Area---Price.png')
plt.show()

plt.figure(figsize=(10,10))
sns.boxplot(x=Data['number of bedrooms'],y=Data['Price'])
plt.title('Rooms---Price')
plt.savefig('Rooms---Price.png')
plt.show()

plt.figure(figsize=(10,10))
sns.boxplot(x=Data['number of bathrooms'],y=Data['Price'])
plt.title('Bathrooms---Price')
plt.savefig('Bathrooms---Price.png')
plt.show()

plt.figure(figsize=(10,10))
sns.scatterplot(x=Data['condition of the house'],y=Data['Price'])
plt.title('Condition---Price')
plt.savefig('Condition---Price.png')
plt.show()


plt.figure(figsize=(10,10))
sns.scatterplot(x=Data['Lattitude'],y=Data['Price'])
plt.title('Location---Price')
plt.savefig('Location---Price.png')
plt.show()


plt.figure(figsize=(10,10))
sns.scatterplot(x=Data['Number of schools nearby'],y=Data['Price'])
plt.title('SchoolDis---Price')
plt.savefig('SchoolDis---Price.png')
plt.show()



plt.figure(figsize=(10,10))
sns.scatterplot(x=Data['Distance from the airport'],y=Data['Price'])
plt.title('AirportDis---Price')
plt.savefig('AirportDis---Price.png')
plt.show()


sns.pairplot(Data[['Price', 'living area', 'number of bedrooms']])
plt.savefig("Pairplot.png")
plt.show()


Data=pd.get_dummies(Data,drop_first=True)

print(Data)


x=Data.drop(columns=['Price'])
y=Data['Price']
pickle.dump(x.columns,open('Columns.pkl','wb'))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


models={
    'linear':LinearRegression(),
    'Poly':Pipeline([('poly',PolynomialFeatures(degree=2)),('linear',LinearRegression())]),
    'Ridge':Ridge(),
    'Lasso':Lasso(),
    'E_net':ElasticNet(l1_ratio=0.5),
    'DecisionTree':DecisionTreeRegressor(random_state=42),
    'GradTree':GradientBoostingRegressor(random_state=42),
    'ADAboost':AdaBoostRegressor(random_state=42),
    'RandomForest':RandomForestRegressor(random_state=42),
    'XGboost':XGBRegressor()
}

results=[]

for name , model in models.items():
    if name in ['DecisionTree','RandomForest','XGboost']:
        model.fit(x_train,y_train)
        prediction=model.predict(x_test)
    else:
        model.fit(x_train_scaled,y_train)
        prediction=model.predict(x_test)
    results.append({
        'Name':name,
        'model':model,
        'MAE':mean_absolute_error(y_test,prediction),
        'MSE':mean_squared_error(y_test,prediction),
        'R2':r2_score(y_test,prediction)*100,
        'RMS':np.sqrt(mean_squared_error(y_test,prediction))
    })

results=pd.DataFrame(results)
print(results.sort_values('R2',ascending=False))


param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5,6,7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_XG=GridSearchCV(XGBRegressor(),cv=5,param_grid=param_grid_xgb,scoring='r2',verbose=2)
grid_XG.fit(x_train,y_train)
best_XG=grid_XG.best_estimator_
best_XG_predict=best_XG.predict(x_test)
print('best parameters XG:',grid_XG.best_params_)
print('best R2 XG:',r2_score(y_test,best_XG_predict))



best_model=XGBRegressor(n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8)

best_model.fit(x_train,y_train)
best_XG_prediction=best_model.predict(x_test)
print(best_XG_prediction,r2_score(y_test,best_XG_prediction)*100)



plt.figure(figsize=(10,10))
plt.scatter(x=y_test,y=best_XG_prediction)
plt.title('Final Plot')
plt.xlabel('Test Data')
plt.ylabel('Prediction Data')
plt.savefig('Test---Predict.png')
plt.show()


importance=best_model.feature_importances_
plt.figure(figsize=(10,10))
plt.barh(range(len(importance)),importance)
plt.savefig('feature_importance.png')
plt.show()


pickle.dump(best_model,open('Model.pkl','wb'))
pickle.dump(scaler,open('Scaler.pkl','wb'))
pickle.dump(x.columns,open('Columns.pkl','wb'))


print('Model Saved ')




