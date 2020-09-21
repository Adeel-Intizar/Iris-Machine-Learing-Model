### Importing Libraries
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/My Drive/iris_model/Iris.csv', index_col=False)

df.drop('Id', axis=1, inplace=True)

df.head()

df.info()

df.describe().T

df.isnull().sum()

setosa = df[df.Species == 'Iris-setosa']
versicolor = df[df.Species == 'Iris-versicolor']
virginica = df[df.Species == 'Iris-virginica']

"""## Visual EDA"""

plt.figure(figsize=(10,6))
plt.style.use('ggplot')
plt.scatter(setosa['SepalLengthCm'], setosa['SepalWidthCm'], c = 'b', label = 'Iris-Setosa')
plt.scatter(versicolor['SepalLengthCm'], versicolor['SepalWidthCm'], c = 'r', label = 'Iris-Versicolor')
plt.scatter(virginica['SepalLengthCm'], virginica['SepalWidthCm'], c = 'm', label = 'Iris-Virginica')
plt.xlabel('Sepal-Length (Cm)', fontsize = 14)
plt.ylabel('Sepal-Width (Cm)', fontsize = 14)
plt.title('Sepal-Length vs Sepal-Width (Cm)', fontsize = 18)
plt.legend(loc = (1.02, 0.8))
plt.show()

plt.figure(figsize=(10,6))
plt.style.use('ggplot')
plt.scatter(setosa['PetalLengthCm'], setosa['PetalWidthCm'], c = 'b', label = 'Iris-Setosa')
plt.scatter(versicolor['PetalLengthCm'], versicolor['PetalWidthCm'], c = 'r', label = 'Iris-Versicolor')
plt.scatter(virginica['PetalLengthCm'], virginica['PetalWidthCm'], c = 'm', label = 'Iris-Virginica')
plt.xlabel('Petal-Length (Cm)', fontsize = 14)
plt.ylabel('Petal-Width (Cm)', fontsize = 14)
plt.title('Petal-Length vs Petal-Width (Cm)', fontsize = 18)
plt.legend(loc = (1.02, 0.8))
plt.show()

df.hist(edgecolor = 'black', linewidth = 1.2)
plt.gcf().set_size_inches(10, 6)
plt.show()

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.violinplot(x='Species', y='SepalLengthCm', data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species', y='SepalWidthCm', data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species', y='PetalLengthCm', data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species', y='PetalWidthCm', data=df)
plt.show()

sns.boxplot(data=df)
plt.show()

df.shape

"""# Building SkLearn Model"""

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

"""## Train Test Split"""

X = df.drop('Species', axis = 1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

"""## Feature Selection"""

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot = True, cmap = 'RdYlGn', square=True)
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.show()

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X, y)
imp_features = pd.Series(model.feature_importances_, index = X.columns).sort_values(ascending=False)
# print(imp_features)
plt.plot(imp_features, color = 'm')
plt.show()

from sklearn.preprocessing import LabelEncoder
g = y.copy()
le = LabelEncoder().fit(g)
# print(le.classes_)
encoded_y = le.transform(g)
from sklearn.linear_model import LassoCV
names = X.columns
lasso = LassoCV(cv=3)
features = lasso.fit(X, encoded_y).coef_
plt.plot(features, color = 'blue')
plt.xticks(range(len(names)), names, rotation = 0)
plt.figure(figsize=(7,4))
plt.show()

"""## Checking Accuracy of Each Model"""

accuracy = []
error = []
classifiers = ['Linear_SVC', 'Radial_SVC', 'KNN', 'Logistic_Regression', 'Decision_Tree', 'Random_Forest']
models = [SVC(gamma='scale', kernel='linear'), SVC(gamma='auto', kernel='rbf'), KNeighborsClassifier(n_neighbors=5), 
         LogisticRegression(solver='liblinear', multi_class='auto'), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_pred, y_test))
    y_pred_encoded = LabelEncoder().fit_transform(y_pred)
    y_test_encoded = LabelEncoder().fit_transform(y_test)
    error.append(np.sqrt(mean_squared_error(y_test_encoded, y_pred_encoded)))
d = {'Accuracy' : accuracy, 'RMSE' : error}
score = pd.DataFrame(d, index = classifiers)
score

"""## Hyper-Parameter Tuning (KNN)"""

param_grid = {'n_neighbors' : np.arange(1, 51)}
knn_model = KNeighborsClassifier()
knn_cv = GridSearchCV(knn_model, param_grid, cv = 5, scoring='accuracy')
knn_cv.fit(X, y)
print(knn_cv.best_params_)
print(knn_cv.best_score_)

neighbors = np.arange(1, 11)
test_accuracy = np.empty(len(neighbors))
train_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    test_accuracy[i] = model.score(X_train, y_train)
    train_accuracy[i] = model.score(X_test, y_test)
plt.plot(train_accuracy, label='Train Accuracy', color = 'c')
plt.plot(test_accuracy, label='Test Accuracy', color = 'm')
plt.title('Checking Overfitting for KNN', fontsize = 18)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend(fontsize=18, loc = (1.05, 0.7))
plt.figure(figsize=(12,6))
plt.show()

"""## Choosing Linear_SVC"""

model = SVC(gamma='auto', kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'Root Mean Square Error: {mean_squared_error(y_test_encoded, y_pred_encoded)}')
print(f'Model Accuracy: {round(accuracy_score(y_test, y_pred), 3) * 100}%')
print('====================================================')
print(f'Here is the "Classification Report"')
print(classification_report(y_test, y_pred))
print(f'\nHere is the "Confusing Matrix"\n')
print(confusion_matrix(y_test, y_pred))
print('====================================================')

"""## Creating pipeline"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
steps = [('model', SVC(kernel='linear', gamma = 'auto'))]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print('{}%'.format(accuracy_score(y_test, y_pred)*100))

"""## Saving and Loading the Model"""

import joblib
#save the model
joblib.dump(pipeline, 'iris_model.pkl')
new_model = joblib.load('iris_model.pkl')
new_model.predict(X_test)
print('{}%'.format(accuracy_score(y_test, y_pred)*100))

"""# Building TensorFlow/Keras Model"""

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('/content/drive/My Drive/iris_model/Iris.csv', index_col=False)
X = df.drop(['Species', 'Id'], axis = 1).values
y = df['Species'].values
y = LabelEncoder().fit_transform(y)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Build the model
model = Sequential()

model.add(Dense(25, input_shape=(4,), activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Adam optimizer with learning rate of 0.001
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

model.fit(X_train, y_train, verbose=0, batch_size=5, epochs=200)

results = model.evaluate(X_test, y_test)

print('Test Loss: {:.2f}'.format(results[0]))
print('Test Accuracy: {:.2f}'.format(results[1]))

model.save('keras_model.h5')

from tensorflow.keras.models import load_model

new_model = load_model('keras_model.h5')

res = new_model.evaluate(X_test, y_test)
print('Test Loss: {:.2f}'.format(res[0]))
print('Test Accuracy: {:.2f}'.format(res[1]))

