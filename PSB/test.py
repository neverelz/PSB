import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Шаг 1: Загрузка тренировочного набора данных
train_data = pd.read_excel("../train.xlsx")

# Шаг 2: Преобразование дат и целевой переменной
train_data['Дата бронирования'] = pd.to_datetime(train_data['Дата бронирования'])
train_data['Дата отмены'] = pd.to_datetime(train_data['Дата отмены'], errors='coerce')
train_data['Заезд'] = pd.to_datetime(train_data['Заезд'])
train_data['Выезд'] = pd.to_datetime(train_data['Выезд'])

# Целевая переменная: отменено бронирование или нет
train_data['Целевое поле'] = train_data['Дата отмены'].notnull().astype('int64')

# Удаление ненужных колонок
train_data.drop(columns=['№ брони', 'Дата отмены', 'Статус брони'], inplace=True)

# One-hot encoding для категориальных переменных
train_data = pd.get_dummies(train_data, drop_first=True)

# Создание признаков
train_data['length_of_stay'] = (train_data['Выезд'] - train_data['Заезд']).dt.days
train_data['time_to_checkin'] = (train_data['Заезд'] - train_data['Дата бронирования']).dt.days
train_data['booking_month'] = train_data['Дата бронирования'].dt.month
train_data['booking_day_of_week'] = train_data['Дата бронирования'].dt.weekday

# Преобразование дат в секунды
train_data['Дата бронирования'] = train_data['Дата бронирования'].astype('int64') // 10**9
train_data['Заезд'] = train_data['Заезд'].astype('int64') // 10**9
train_data['Выезд'] = train_data['Выезд'].astype('int64') // 10**9

# Разделение на признаки и целевую переменную
X_train = train_data.drop(columns=['Целевое поле'])
y_train = train_data['Целевое поле']

# Инициализация модели с лучшими гиперпараметрами
best_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=1,
    random_state=0
)

# Обучение модели
best_model.fit(X_train, y_train)

# Оценка ROC AUC на обучающей выборке
y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
roc_auc_train = roc_auc_score(y_train, y_train_pred_proba)
print(f"ROC AUC на обучающей выборке: {roc_auc_train:.4f}")

# Шаг 3: Работа с новыми тестовыми данными
test_data = pd.read_excel("../test.xlsx")

# Преобразование дат и признаков
test_data['Дата бронирования'] = pd.to_datetime(test_data['Дата бронирования'])
test_data['Заезд'] = pd.to_datetime(test_data['Заезд'])
test_data['Выезд'] = pd.to_datetime(test_data['Выезд'])

# Создание признаков для тестовых данных
test_data['length_of_stay'] = (test_data['Выезд'] - test_data['Заезд']).dt.days
test_data['time_to_checkin'] = (test_data['Заезд'] - test_data['Дата бронирования']).dt.days
test_data['booking_month'] = test_data['Дата бронирования'].dt.month
test_data['booking_day_of_week'] = test_data['Дата бронирования'].dt.weekday

# Преобразование дат в секунды
test_data['Дата бронирования'] = test_data['Дата бронирования'].astype('int64') // 10**9
test_data['Заезд'] = test_data['Заезд'].astype('int64') // 10**9
test_data['Выезд'] = test_data['Выезд'].astype('int64') // 10**9

# One-hot encoding для категориальных переменных
test_data = pd.get_dummies(test_data, drop_first=True)

# Приведение нового набора данных к той же структуре, что и тренировочный
missing_cols = set(X_train.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[X_train.columns]

# Шаг 4: Предсказание для нового набора данных
predictions = best_model.predict(test_data)

# Сохранение результата
test_data['Предсказания'] = predictions
test_data[['Предсказания']].to_csv('../new_predictions.csv', index=False, header=False)

print("Предсказания для нового набора данных сохранены в файл")
