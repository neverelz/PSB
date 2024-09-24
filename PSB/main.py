import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Шаг 1: Загрузка тренировочного и тестового набора данных
train_data = pd.read_excel("../train.xlsx")
test_data = pd.read_excel("../test.xlsx")

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

# Создание признаков "длительность пребывания", "time_to_checkin" и других
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

# Обучение модели
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

# Получение важности признаков
feature_importances = model.feature_importances_

# Создание DataFrame для визуализации
importances_df = pd.DataFrame({'Признак': X_train.columns, 'Важность': feature_importances})

# Сортировка признаков по важности
importances_df = importances_df.sort_values(by='Важность', ascending=False)

important_features = importances_df[importances_df['Важность'] > 0.01]['Признак'].tolist()

# Шаг 3: Преобразование тестового набора данных

# Преобразование дат
test_data['Дата бронирования'] = pd.to_datetime(test_data['Дата бронирования'])
test_data['Заезд'] = pd.to_datetime(test_data['Заезд'])
test_data['Выезд'] = pd.to_datetime(test_data['Выезд'])

# Создание признаков "длительность пребывания", "time_to_checkin" и других
test_data['length_of_stay'] = (test_data['Выезд'] - test_data['Заезд']).dt.days
test_data['time_to_checkin'] = (test_data['Заезд'] - test_data['Дата бронирования']).dt.days
test_data['booking_month'] = test_data['Дата бронирования'].dt.month
test_data['booking_day_of_week'] = test_data['Дата бронирования'].dt.weekday

# Преобразование дат в секунды
test_data['Дата бронирования'] = test_data['Дата бронирования'].astype('int64') // 10**9
test_data['Заезд'] = test_data['Заезд'].astype('int64') // 10**9
test_data['Выезд'] = test_data['Выезд'].astype('int64') // 10**9

# One-hot encoding для категориальных переменных в тестовом наборе
test_data = pd.get_dummies(test_data, drop_first=True)

# Приведение тестового набора к той же структуре, что и тренировочный
# Добавляем недостающие колонки
missing_cols = set(X_train.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0

# Удаляем лишние колонки, которые не присутствуют в тренировочном наборе
extra_cols = set(test_data.columns) - set(X_train.columns)
test_data.drop(columns=extra_cols, inplace=True)

# Переводим тестовые данные в тот же порядок колонок, что и в тренировочном наборе
test_data = test_data[X_train.columns]

# Шаг 4: Предсказание
predictions = model.predict(test_data)


# Сохранение результата
test_data['Предсказания'] = predictions
test_data[['Предсказания']].to_csv('../new_predictions.csv', index=False, header=False)

print("Предсказания для нового набора данных сохранены в файл")

# Чтение предсказаний и вычисление суммы
predictions = pd.read_csv('../new_predictions.csv', header=None, names=['Предсказания'])
sum_predictions = predictions['Предсказания'].sum()

print(f"Сумма предсказаний: {sum_predictions}")
