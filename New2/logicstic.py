import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("User_Data.csv")

x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

print (xtrain[0:10, :])

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)

from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(ytest, y_pred))

from matplotlib.colors import ListedColormap

X_set, y_set = xtest, ytest
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                               stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                               stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(
    np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



""
Logistic regression is a type of statistical analysis that is used to model the relationship between a binary dependent variable and one or more independent variables. 
In other words, it is a method for predicting the probability of a certain outcome, based on a set of input variables.
The logistic regression model uses a mathematical function called the logistic function, which is a type of S-shaped curve, to estimate the probability of the dependent variable taking a particular value. 
The model calculates a weighted sum of the input variables, and then applies the logistic function to that sum to obtain a predicted probability.
Logistic regression is often used in binary classification problems, such as predicting whether a customer will buy a product or not, or whether a patient will have a certain medical condition or not. 
It can also be used in multiclass classification problems by applying the model to each class separately.
Logistic regression is a popular and widely-used method in machine learning and data analysis, due to its simplicity, interpretability, and effectiveness in many practical applications. 
It is also a fundamental building block for more complex machine learning algorithms, such as neural networks.
""