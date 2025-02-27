import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder  # OneHotEncoder 임포트 추가
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# mpg 데이터셋 로드
mpg = sns.load_dataset('mpg')

# 결측값 처리: horsepower 열에서 결측값 제거
mpg.dropna(subset=["horsepower"], inplace=True)

# 'name' 열에서 첫 번째 단어만 추출하여 새로운 'company' 열에 저장
mpg['company'] = mpg['name'].str.split().str[0]
mpg.drop("name", axis=1, inplace=True)  # 'name' 열 삭제

# 'origin'과 'company' 열에 OneHotEncoder 적용
origin_encoder = OneHotEncoder(sparse_output=False)  # sparse_output=True로 설정하여 배열 형태로 반환
company_encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False로 설정하여 배열 형태로 반환

# 'origin' 열 OneHotEncoding
origin_cat_1hot = origin_encoder.fit_transform(mpg[['origin']])

# 'company' 열 OneHotEncoding
company_cat_1hot = company_encoder.fit_transform(mpg[['company']])

# OneHotEncoding 결과를 데이터프레임으로 변환하여 열 이름을 설정
origin_cat_1hot_df = pd.DataFrame(origin_cat_1hot, columns=origin_encoder.get_feature_names_out(['origin']), index=mpg.index)
company_cat_1hot_df = pd.DataFrame(company_cat_1hot, columns=company_encoder.get_feature_names_out(['company']), index=mpg.index)

# 기존 mpg 데이터프레임에서 'origin'과 'company' 열 삭제하고, 인코딩된 열들을 합침
mpg = pd.concat([mpg.drop(['origin', 'company'], axis=1), origin_cat_1hot_df, company_cat_1hot_df], axis=1)

# 독립변수(X)와 종속변수(y) 설정
X = mpg.drop(['mpg'], axis=1)  # 레이블 컬럼 제거
y = mpg['mpg']  # 레이블 (타겟 변수)

# 훈련 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 결과 출력
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title('Actual MPG vs Predicted MPG')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.grid(True)
plt.show()
