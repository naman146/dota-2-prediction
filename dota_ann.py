import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

#IMPORTING THE DATASET
df = pd.read_csv('dota_games.csv',header=None)
df = df.rename(columns={0: 'ancient_1', 1: 'ancient_2', 2: 'ancient_3', 3: 'ancient_4', 4: 'ancient_5',
                       5: 'dire_1', 6: 'dire_2', 7: 'dire_3', 8: 'dire_4', 9: 'dire_5', 10: 'team_win'})

#CHECKING NULL VALUEs IF ANY
df.isnull().values.any()

y = df['team_win']
X = df.drop(['team_win'], axis=1)
for val in range(0,15000):
    if y[val] == 1:
        y[val] = 0
    else:
        y[val] = 1
#
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label = LabelEncoder()
for col in X.columns.values:
    X[col] = label.fit_transform(X[col])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
#SPLITTING INTO TEST AND TRAIN SET

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


from sklearn.preprocessing import StandardScaler
standscale = StandardScaler()
X_train = standscale.fit_transform(X_train)
X_test = standscale.transform(X_test)

#ANN
classifier= Sequential()
classifier.add(Dense(input_dim = 105, units=53, activation = 'relu'))
classifier.add(Dense(units = 53, activation = 'relu'))
#classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy',metrics = ['accuracy'])

history = classifier.fit(X_train, y_train, batch_size =10, epochs = 20)


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
