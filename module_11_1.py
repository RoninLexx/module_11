#Выбранные библиотеки: requests, pandas, matplotlib


#requests


#* [Документация](https://requests.readthedocs.io/en/latest/)
#* Возможности:
#* Отправка HTTP-запросов
#* Получение и обработка ответов
#* Работа с файлами cookie и сессиями
#* Функции:
#* get(url): отправка GET-запроса
#* post(url, data): отправка POST-запроса
#* headers: получение и установка заголовков запроса

#Пример кода:


import requests

# Отправка GET-запроса
response = requests.get("https://www.example.com")

# Получение тела ответа
print(response.text)

# 1. Выполнить GET-запрос
response = requests.get('https://jsonplaceholder.typicode.com/posts')
print("Status Code:", response.status_code)
print("Response JSON:", response.json()[:2])  # Выводим первые 2 поста

# 2. Выполнить POST-запрос
payload = {'title': 'foo', 'body': 'bar', 'userId': 1}
post_response = requests.post('https://jsonplaceholder.typicode.com/posts', json=payload)
print("Created Post ID:", post_response.json()['id'])

# 3. Добавить заголовки к запросу
headers = {'Authorization': 'Bearer YOUR_ACCESS_TOKEN'}
auth_response = requests.get('https://jsonplaceholder.typicode.com/posts', headers=headers)
print("Auth Response Status Code:", auth_response.status_code)


##pandas


#* [Документация](https://pandas.pydata.org/docs/)
#* Возможности:
#* Загрузка и обработка данных из различных источников
#* Манипулирование и анализ данных
#* Визуализация данных
#* Функции/классы:
#* read_csv(filename): считывание данных из CSV-файла
#* DataFrame: структура данных для представления табличных данных
#* mean(): вычисление среднего значения столбца




import pandas as pd

# 1. Создание DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
print("DataFrame:\n", df)

# 2. Сохранение DataFrame в CSV-файл
df.to_csv("data.csv", index=False)  # Сохраняем данные в файл

# 3. Чтение данных из CSV файла
df_from_csv = pd.read_csv("data.csv")

# 4. Вычисление среднего значения столбца "Age"
mean_age = df_from_csv["Age"].mean() 

# 5. Вывод результатов
print(f"Средний возраст: {mean_age}")


#matplotlib


#* [Документация](https://matplotlib.org/)
#* Возможности:
#* Создание статических, анимированных и интерактивных визуализаций
#* Поддержка различных типов графиков и диаграмм
#* Настройка внешнего вида и параметров графиков
#* Классы:
#* Figure: холст для рисования
#* Axes: область рисования на холсте
#* pyplot: модуль для упрощенного создания графиков




import matplotlib.pyplot as plt

# Создание графика
plt.plot([1, 2, 3, 4], [5, 6, 7, 8])

# Настройка внешнего вида графика
plt.title("График")
plt.xlabel("x")
plt.ylabel("y")

# Отображение графика
plt.show()

# 1. Простой линейный график
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
plt.plot(x, y, label='Prime Numbers', marker='o')
plt.title('Simple Line Plot')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()

# 2. Гистограмма
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5]
plt.hist(data, bins=5, alpha=0.7, color='blue')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# 3. Столбчатая диаграмма
categories = ['A', 'B', 'C']
values = [10, 20, 15]
plt.bar(categories, values, color='orange')
plt.title('Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()
