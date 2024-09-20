import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Step 1: Load the data
data = pd.read_excel("../train.xlsx")

# Step 2: Data Preprocessing
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

# Convert datetime columns to numerical features (e.g., timestamps)
data['Дата бронирования'] = data['Дата бронирования'].astype(int) // 10**9  # Convert to seconds
data['Заезд'] = data['Заезд'].astype(int) // 10**9  # Convert to seconds
data['Выезд'] = data['Выезд'].astype(int) // 10**9  # Convert to seconds

# Step 4: Train the model
X = data.drop(columns=['Целевое поле'])
y = data['Целевое поле']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'ROC-AUC Score: {roc_auc}')

# Save the output
output = pd.DataFrame({'Predictions': y_pred})
output.to_csv('../answer.csv', header=False, index=False, sep=',')
