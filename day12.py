import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Titanic 데이터셋 로드
titanic = sns.load_dataset('titanic')

# 'age'가 NaN인 사람들의 나이를 죽은 사람들의 평균 나이로 채우기
mean_age_of_dead = titanic[titanic['survived'] == 0]['age'].mean()
titanic['age'] = titanic['age'].fillna(mean_age_of_dead)

X = titanic[['age']]
y = titanic['survived']

# 결과 확인
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(5, 2))
plt.scatter(X_test, y_test, color='blue', label='Real')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('KNeighborsClassifier Real vs Predicted')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.show()