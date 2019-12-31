import pandas as pd
import numpy as np
import keras


#IMPORTING THE DATASET
df = pd.read_csv('dota_games.csv',header=None)
df = df.rename(columns={0: 'ancient_1', 1: 'ancient_2', 2: 'ancient_3', 3: 'ancient_4', 4: 'ancient_5',
                       5: 'dire_1', 6: 'dire_2', 7: 'dire_3', 8: 'dire_4', 9: 'dire_5', 10: 'team_win'})

#CHECKING NULL VALUEs IF ANY
df.isnull().values.any()

X = df.iloc[:, 0:10]
y = df.iloc[:, 10]

#
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
for col in X.columns.values:
    X[col] = label.fit_transform(X[col])

from sklearn.preprocessing import StandardScaler
standscale = StandardScaler()
X = standscale.fit_transform(X)

#SPLITTING INTO TEST AND TRAIN SET

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#SVM

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


acc=0.526
https://www.kaggle.com/renanmav/dota-2-game-prediction