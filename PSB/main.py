import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# Создание признака "длительность пребывания"
train_data['length_of_stay'] = (train_data['Выезд'] - train_data['Заезд']).dt.days

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

print("Модель обучена на тренировочном наборе данных")

test_data = pd.read_excel("../test.xlsx")

# Шаг 4: Преобразование дат и признаков
test_data['Дата бронирования'] = pd.to_datetime(test_data['Дата бронирования'])
test_data['Заезд'] = pd.to_datetime(test_data['Заезд'])
test_data['Выезд'] = pd.to_datetime(test_data['Выезд'])

# Создание признака "длительность пребывания"
test_data['length_of_stay'] = (test_data['Выезд'] - test_data['Заезд']).dt.days

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

# Шаг 5: Предсказание
predictions = model.predict(test_data)

# Сохранение результата
test_data['Предсказания'] = predictions
test_data[['Предсказания']].to_csv('../new_predictions.csv', index=False, header=False)

print("Предсказания для нового набора данных сохранены в файл")


