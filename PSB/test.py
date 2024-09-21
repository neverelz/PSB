import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_excel('../test.xlsx')
df = pd.DataFrame(data)

# Преобразуем колонки с датами в формат datetime
df["Дата бронирования"] = pd.to_datetime(df["Дата бронирования"])
df["Заезд"] = pd.to_datetime(df["Заезд"])
df["Выезд"] = pd.to_datetime(df["Выезд"])

# Добавим новые признаки: разница между датой бронирования и заезда
df["Разница до заезда (дни)"] = (df["Заезд"] - df["Дата бронирования"]).dt.days
df["Продолжительность пребывания (дни)"] = (df["Выезд"] - df["Заезд"]).dt.days

df.to_csv('../arrivals.csv', sep=',')

# Анализ скрытых зависимостей через корреляции числовых переменных
corr_matrix = df[["Номеров", "Стоимость", "Внесена предоплата", "Ночей", "Гостей", "Гостиница", "Разница до заезда (дни)", "Продолжительность пребывания (дни)"]].corr()

# Визуализация корреляционной матрицы
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Корреляционная матрица')
plt.xticks(rotation=45)
plt.show()

# Анализ зависимостей категориальных переменных
# Например, зависимость между формой оплаты и количеством дней до заезда
sns.boxplot(x="Способ оплаты", y="Разница до заезда (дни)", data=df)
plt.title('Разница до заезда в зависимости от способа оплаты')
plt.xticks(rotation=45)
plt.show()


