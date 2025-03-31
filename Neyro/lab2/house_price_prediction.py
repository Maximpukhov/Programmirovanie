# house_price_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Загрузка данных
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Анализ данных
print(train_data.info())
print(train_data.describe())

# Визуализация распределения целевой переменной
plt.figure(figsize=(10, 6))
sns.histplot(train_data['SalePrice'], kde=True)
plt.title("Распределение цен на дома")
plt.savefig('price_distribution.png')  # Сохранение графика
plt.show()

# Корреляция числовых признаков с SalePrice
numeric_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = train_data[numeric_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix[['SalePrice']].sort_values(by='SalePrice', ascending=False), annot=True)
plt.title("Корреляция признаков с ценой дома")
plt.savefig('correlation_heatmap.png')
plt.show()

# Предобработка данных
# Разделение на числовые и категориальные признаки
num_cols = train_data.select_dtypes(include=['int64', 'float64']).drop('SalePrice', axis=1).columns
cat_cols = train_data.select_dtypes(include=['object']).columns

# Создание пайплайна для предобработки
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# Разделение данных
X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Визуализация предсказаний vs реальных значений
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Предсказания vs Реальные значения')
plt.savefig('predictions_vs_actual.png')
plt.show()

# Важность признаков (для tree-based моделей)
if hasattr(model.named_steps['regressor'], 'feature_importances_'):
    feature_importances = model.named_steps['regressor'].feature_importances_
    
    # Получение имен признаков после OneHotEncoder
    onehot_columns = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    all_features = np.concatenate([num_cols, onehot_columns])
    
    # Создание DataFrame с важностью признаков
    importance_df = pd.DataFrame({'Feature': all_features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
    
    # Визуализация
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Топ-20 важных признаков')
    plt.savefig('feature_importance.png')
    plt.show()

    # Предсказание на тестовых данных (если нужно)
if not test_data.empty:
    test_predictions = model.predict(test_data)
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'SalePrice': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("Предсказания для тестовых данных сохранены в submission.csv")
