import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test,y_pred)

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

"""
SVM stands for Support Vector Machine, and it is a popular algorithm used in machine learning for classification and regression analysis. 
SVM is a supervised learning model that can be used for both linear and nonlinear classification, as well as regression tasks.
The basic idea behind SVM is to find the best hyperplane that separates two classes in the input space. 
In other words, it tries to maximize the margin between the two classes. 
The margin is defined as the distance between the hyperplane and the nearest data points from each class. 
The data points that lie closest to the hyperplane are called support vectors.
There are different variations of SVMs, such as linear SVM, kernel SVM, and multiclass SVM. 
Linear SVM is used when the data is linearly separable, whereas kernel SVM is used when the data is nonlinearly separable. 
Multiclass SVM is used when there are more than two classes in the data.
SVMs have several advantages, such as their ability to handle high-dimensional data, their robustness to noise, and their generalization performance. 
However, they can be computationally expensive, especially when dealing with large datasets.
Overall, SVMs are a powerful tool in the field of machine learning, and they have been successfully applied to various real-world problems, such as image classification, text classification, and bioinformatics.
"""