# %%
import pandas as pd
import sklearn
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from definitions import ROOT_DIR

# %%
train_df = pd.read_csv(f'{ROOT_DIR}/data/train.csv')
test_df = pd.read_csv(f'{ROOT_DIR}/data/test.csv')

# %%
# Replace NaN values for Age and Fare with the mean
train_df['Age'].fillna(train_df['Age'].median(),inplace=True)
test_df['Age'].fillna(train_df['Age'].median(),inplace=True)

# Replace missing NaN values for Fare in testing data
test_df['Fare'].fillna(test_df['Fare'].median(),inplace=True)

# Replace Embarked with most frequent value
train_df['Embarked'].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)

# %%
# Extract the Cabin Letter from Cabin and replace NaN values with 'U' for unknown value
train_df['CabinLetter']=train_df['Cabin'].str.slice(0,1)
train_df['CabinLetter'].fillna('U', inplace=True)
test_df['CabinLetter']=train_df['Cabin'].str.slice(0,1)
test_df['CabinLetter'].fillna('U', inplace=True)

# Replace one T value with U
train_df['CabinLetter'].replace(['T'], ['U'], inplace=True)
test_df['CabinLetter'].replace(['T'], ['U'], inplace=True)

# %%
# Match char "," followed by whitespace "\s" zero or one times then capture and match any char one or more occurrences (.+?) then char "." then match whitespace "\s"
train_df['Title'] = train_df['Name'].str.extract(r',\s?(.+?)\.\s')
test_df['Title'] = test_df['Name'].str.extract(r',\s?(.+?)\.\s')

# %%
# Replace titles to more common gender associated titles
train_df['Title'].replace(['Ms', 'Lady', 'Mme', 'Mlle', 'Dona', 'the Countess'], ['Miss', 'Miss', 'Miss', 'Miss', 'Mrs', 'Mrs'], inplace=True)
train_df['Title'].replace(['Sir', 'Capt', 'Col', 'Jonkheer', 'Don', 'Major'], ['Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mr'], inplace=True)
test_df['Title'].replace(['Mme', 'Dona', 'Ms'], ['Miss', 'Mrs', 'Miss'], inplace=True)
test_df['Title'].replace(['Don', 'Col'], ['Mr', 'Mr'], inplace=True)

# %%
Y_train = train_df[['Survived']].copy()
X_train_df = train_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','CabinLetter']].copy()
X_test_df = test_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','CabinLetter']].copy()

# %%
# Transform into Numerical Values using pd.get_dummies(drop_first=True) to get k-1 dummies out of k categorical levels by removing the first level.
X_train=pd.get_dummies(data=X_train_df, columns=['Sex', 'Embarked', 'Title', 'CabinLetter'], drop_first=True)
X_test=pd.get_dummies(data=X_test_df, columns=['Sex', 'Embarked', 'Title', 'CabinLetter'], drop_first=True)

# %%
# scaling
scale=StandardScaler().fit(X_train)
X_train_sc=scale.transform(X_train)
X_test_sc=scale.transform(X_test)

# %%
svc = SVC()
svc.fit(X_train_sc, Y_train.values.ravel())
Y_pred = svc.predict(X_test_sc)
svc.score(X_train_sc, Y_train)

# %%
submission = pd.DataFrame({"PassengerId":test_df["PassengerId"], "Survived": Y_pred})
submission.to_csv(f"{ROOT_DIR}/submission/submission.csv", index=False)