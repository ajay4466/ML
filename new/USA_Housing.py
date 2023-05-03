import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

HouseDF = pd.read_csv('USA_Housing.csv')
HouseDF.head()

HouseDF.info()


HouseDF.describe()


HouseDF.columns


sns.pairplot(HouseDF)


sns.distplot(HouseDF['Price'])


sns.heatmap(HouseDF.corr(), annot=True)


X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = HouseDF['Price']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


from sklearn.linear_model import LinearRegression 
lm = LinearRegression() 
lm.fit(X_train,y_train) 

print(lm.intercept_)

predictions = lm.predict(X_test)  
plt.scatter(y_test,predictions)


sns.distplot((y_test-predictions),bins=50); 

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient']) coeff_df ## bt huva he to sabhal le na



