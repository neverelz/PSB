import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

# Шаг 1: Загрузка данных
file_path = '../train.xlsx'  # Укажите путь к вашему файлу
data = pd.read_excel(file_path)

# Шаг 2: Обработка данных
data['cancellation'] = data['Дата отмены'].notnull().astype(int)

X = data.drop(['№ брони', 'Дата бронирования', 'Дата отмены', 'Заезд', 'Выезд', 'cancellation'], axis=1)
y = data['cancellation']

# One-Hot Encoding для категориальных признаков
categorical_features = ['Способ оплаты', 'Источник', 'Статус брони', 'Категория номера', 'Гостиница']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_categorical = encoder.fit_transform(X[categorical_features])

encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out())
X = X.drop(categorical_features, axis=1)
X = pd.concat([X.reset_index(drop=True), encoded_categorical_df.reset_index(drop=True)], axis=1)

# Масштабирование числовых признаков
scaler = StandardScaler()
numerical_features = ['Номеров', 'Стоимость', 'Ночей', 'Гостей']
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Шаг 3: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 4: Обучение модели случайного леса с подбором гиперпараметров
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Лучшая модель случайного леса
best_rf_model = grid_search.best_estimator_

# Шаг 5: Предсказания и оценка случайного леса
y_pred_rf = best_rf_model.predict(X_test)

print("\nСлучайный лес:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf)}")
print(f"Recall: {recall_score(y_test, y_pred_rf)}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf)}")

# Шаг 6: Сохранение предсказаний в CSV файл
results_df = pd.DataFrame({'Predictions': y_pred_rf})
results_df.to_csv('../answer.csv', header=False, index=False)
