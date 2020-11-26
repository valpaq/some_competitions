import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.shape, test.shape)

X_train = train.drop('Culture', axis=1)
y_train = train['Culture']

X_test = test.copy()

print(X_train.shape, X_test.shape, y_train.shape)


imputer = KNNImputer(n_neighbors=9, weights="distance")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

X_tt, X_tv, y_tt, y_tv = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

model = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=15, max_features = 'sqrt')
model.fit(X_tt, y_tt)
preds_valid = model.predict(X_tv)

print(f1_score(y_tv, preds_valid, average='weighted'))

preds = model.predict(X_test) + 1

pd.Series(preds).to_csv('preds.csv', index=False, header=['Culture'])