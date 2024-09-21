import pandas as pd

data = pd.read_excel('../train.xlsx')

data['Заезд'] = pd.to_datetime(data['Заезд'])

# Распределение по гостиницам и формам оплаты
payment_methods_by_hotel = data.groupby('Гостиница')['Способ оплаты'].value_counts(normalize=True).unstack()
payment_methods_by_hotel.to_csv('payment_methods_by_hotel.csv', encoding='utf-8')


# Распределение по датам заезда
data['Месяц заезда'] = data['Заезд'].dt.month
arrival_by_month = data.groupby(['Гостиница', 'Месяц заезда']).size().unstack()
arrival_by_month.to_csv('arrival_by_month.csv', encoding='utf-8')
