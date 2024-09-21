import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


data = pd.read_excel("../train.xlsx")


# Convert dates to datetime
data['Дата бронирования'] = pd.to_datetime(data['Дата бронирования'])
data['Дата отмены'] = pd.to_datetime(data['Дата отмены'], errors='coerce')
data['Заезд'] = pd.to_datetime(data['Заезд'])
data['Выезд'] = pd.to_datetime(data['Выезд'])

# Create target variable
data['Целевое поле'] = data['Дата отмены'].notnull().astype(int)

# Drop unnecessary columns
data.drop(columns=['№ брони', 'Дата отмены', 'Статус брони'], inplace=True)

# Handle categorical variables
data = pd.get_dummies(data, drop_first=True)

# Step 3: Feature Engineering
# Example: Create a feature for the length of stay
data['length_of_stay'] = (data['Выезд'] - data['Заезд']).dt.days


data['Дата бронирования'] = data['Дата бронирования'].astype(int) // 10**9  # Convert to seconds
data['Заезд'] = data['Заезд'].astype(int) // 10**9  # Convert to seconds
data['Выезд'] = data['Выезд'].astype(int) // 10**9  # Convert to seconds


X = data.drop(columns=['Целевое поле'])
y = data['Целевое поле']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=0)

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

X_full = data.drop(columns=['Целевое поле'])
full_predictions = model.predict(X_full)
data['Предсказания'] = full_predictions

# Save the output
output = pd.DataFrame({'Предсказания': full_predictions})
output.to_csv('../answer.csv', index=False, header=False, sep=',')

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'ROC-AUC Score: {roc_auc}')