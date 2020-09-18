
"""# **Scikit-learn Models**"""

from sklearn.svm import SVC
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('iris.csv')
df.drop('Id', inplace=True, axis=1)
X = df.drop('Species', axis=1, inplace=False)
y = df['Species']
z = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, z, test_size= 0.3, random_state= 42, stratify = y)


model = SVC(kernel='linear', gamma=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy of Linear_SVC: {:.0f}%'.format(accuracy_score(y_test, y_pred) * 100))

if dump(model, 'sklearn_model.pkl'):
    print(True)

"""# **Keras Models**"""

# from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
y_encoded = encoder.fit_transform(z)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify= y_encoded)

def model():
  model = Sequential()
  model.add(Dense(16, activation='relu', input_dim=4))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(3, activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  return model


model = model()
model.fit(X_train, y_train, epochs=300, shuffle=True, verbose=0)

performance = model.evaluate(X_test, y_test, verbose=0)
print('{} : {:.0f}% '.format(model.metrics_names[1], performance[1] * 100))
print('{} : {:.0f}% '.format(model.metrics_names[0], performance[0] * 100))

pred_y = model.predict_classes(X_test)
true_y = encoder.inverse_transform(y_test)
error = (pred_y != true_y).sum()
print('No of Wrong Predictions: {:.0f}%'.format(error * 100))

model.save('keras_model.h5')