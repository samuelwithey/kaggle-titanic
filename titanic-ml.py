# %%
import pandas as pd
import sklearn
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import definitions
from definitions import ROOT_DIR

# %%
train_df = pd.read_csv(f'{ROOT_DIR}/data/train.csv')
test_df = pd.read_csv(f'{ROOT_DIR}/data/test.csv')

# %%
"""
**Data Preprocessing**
"""

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
# Calculate boolean Family column if passenger had family members
train_df['Family'] = train_df["Parch"] + train_df["SibSp"]
train_df['Family'] = np.where(train_df['Family'] > 0, 1, 0)
test_df['Family'] = test_df["Parch"] + test_df["SibSp"]
test_df['Family'] = np.where(test_df['Family'] > 0, 1, 0)

# %%
Y_train = train_df[['Survived']].copy()
X_train_df = train_df[['Pclass','Sex','Age','Fare','Embarked','Title','CabinLetter', 'Family']].copy()
X_test_df = test_df[['Pclass','Sex','Age','Fare','Embarked','Title','CabinLetter', 'Family']].copy()

# %%
# Transform into Numerical Values using pd.get_dummies(drop_first=True) to get k-1 dummies out of k categorical levels by removing the first level.
X_train=pd.get_dummies(data=X_train_df, columns=['Embarked', 'Title', 'CabinLetter'], drop_first=True)
X_test=pd.get_dummies(data=X_test_df, columns=['Embarked', 'Title', 'CabinLetter'], drop_first=True)
X_train['Sex'].replace(['male','female'],[1,0],inplace=True)
X_test['Sex'].replace(['male','female'],[1,0],inplace=True)

# %%
"""
**Grid Search and Modelling**
"""

# %%
# Scaling
scale=StandardScaler().fit(X_train)
X_train_sc=scale.transform(X_train)
X_test_sc=scale.transform(X_test)

# %%
# Support-Vector Machine
parameters = {'probability':[True], 'C':np.linspace(1,1000, num = 10), 'gamma':np.linspace(0.1,1,num=1)}
gs_svc = GridSearchCV(SVC(),param_grid = parameters, scoring="accuracy",n_jobs=-1)
gs_svc.fit(X_train_sc, Y_train.values.ravel())
best_svc = gs_svc.best_estimator_
print(best_svc)
print(gs_svc.best_params_)
print('score=',gs_svc.best_score_)

# %%
# Decision Tree
parameters={'max_features': [1, 2, 3, 5, 6, 7, 8, 9, 10, 15],
          'min_samples_split': [2, 3, 4, 5, 6, 7, 10, 15],
          'min_samples_leaf': [1, 2, 3, 5, 6, 7, 8, 10, 15],
          'splitter':['best']}

gs_dt = GridSearchCV(DecisionTreeClassifier(),param_grid = parameters, scoring="accuracy",n_jobs=-1)
gs_dt.fit(X_train_sc,Y_train.values.ravel())
best_dt=gs_dt.best_estimator_
print(best_dt)
print(gs_dt.best_params_)
print('score=',gs_dt.best_score_)

# %%
# Random Forest
parameters ={'max_features': [1, 2, 3, 5, 10],
          'min_samples_split': [2, 3, 5, 7, 10],
          'min_samples_leaf': [1, 3, 5, 7, 10],
          'bootstrap': [False],
          'n_estimators' :[100,200,300]}

gs_rf = GridSearchCV(RandomForestClassifier(),param_grid = parameters, scoring="accuracy",n_jobs=-1)
gs_rf.fit(X_train_sc,Y_train.values.ravel())
best_rf=gs_rf.best_estimator_
print(best_rf)
print(gs_rf.best_params_)
print('score=',gs_rf.best_score_)

# %%
vote=VotingClassifier(estimators=[('rfc',best_rf),
                                  ('svc',best_svc),
                                  ('dtc',best_dt)],
                      voting='soft', n_jobs=-1)

vote = vote.fit(X_train_sc, Y_train.values.ravel())

# %%
"""
**Prediction**
"""

# %%
Y_predict=vote.predict(X_test_sc)

# %%
"""
**Submission and CSV download**
"""

# %%
submission = pd.DataFrame({"PassengerId":test_df["PassengerId"], "Survived": Y_predict})
submission.to_csv(f"{ROOT_DIR}/submission/submission.csv", index=False)

# %%
