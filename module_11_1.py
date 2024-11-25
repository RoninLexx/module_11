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

import numpy as np

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)

print(a.ndim)  # Количество измерений массива

print(a.size)  # Количество элементов в массиве

print(a.shape)  # Размерность массива

print(a.dtype)  # Тип данных в массиве

np.zeros(2)  # c 0 элементами
np.ones(2)  # Пустой массив с 1 элементом
np.empty(2)  # пустого массива
np.eye(2)  # Создание матрицы
np.random.random(2)  # Массив случайных чисел
np.random.randint(1, 10, 2)  # Массив случайных чисел
np.random.randn(2)  # Массив случайных чисел
np.random.rand(2)  # Массив случайных чисел
np.random.seed(100)  # Установка начального значения для генератора случайных чисел
x = np.ones(2, dtype=np.int64)  # Создание массива b указать нужный тип данных с помощью dtype

np.linspace(0, 10, 5)  # Список чисел от 0 до 10 с шагом 2
np.arange(2, 9, 2)  # Список чисел от 2 до 9 включительно с шагом 2

print(a.itemsize)  # Размер элемента в байтах

print(a.nbytes)  # Размер массива в байтах

a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.shape)  # Форма массива — это кортеж из неотрицательных целых чисел,
# которые определяют количество элементов в каждом измерении.

arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
print(arr)  # Вывод массива
print(np.sort(arr))  # Возвращает отсортированную копию массива
print(np.argsort(arr))  # Возвращает индексы, по которым можно было бы отсортировать массив.
print(np.argmax(arr))  # Возвращает индекс максимального значения вдоль оси.
print(np.argmin(arr))  # Возвращает индексы минимальных значений вдоль оси.
print(np.mean(arr))  # среднее арифметическое
print(np.median(arr))  # медиану вдоль указанной оси.
print(np.std(arr))  # стандартное отклонение вдоль указанной оси.
print(np.var(arr))  # отклонение вдоль указанной оси.
print(np.sum(arr))  # Сумма элементов массива по заданной оси
print(np.cumsum(arr))  # Возвращает совокупную сумму элементов вдоль заданной оси
print(np.cumprod(arr))  # Возвращает суммарное произведение элементов вдоль заданной оси.

import numpy as np

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)

print(a.ndim)  # Количество измерений массива

print(a.size)  # Количество элементов в массиве

print(a.shape)  # Размерность массива

print(a.dtype)  # Тип данных в массиве

np.zeros(2)  # c 0 элементами
np.ones(2)  # Пустой массив с 1 элементом
np.empty(2)  # пустого массива
np.eye(2)  # Создание матрицы
np.random.random(2)  # Массив случайных чисел
np.random.randint(1, 10, 2)  # Массив случайных чисел
np.random.randn(2)  # Массив случайных чисел
np.random.rand(2)  # Массив случайных чисел
np.random.seed(100)  # Установка начального значения для генератора случайных чисел
x = np.ones(2, dtype=np.int64)  # Создание массива b указать нужный тип данных с помощью dtype

np.linspace(0, 10, 5)  # Список чисел от 0 до 10 с шагом 2
np.arange(2, 9, 2)  # Список чисел от 2 до 9 включительно с шагом 2

print(a.itemsize)  # Размер элемента в байтах

print(a.nbytes)  # Размер массива в байтах

a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.shape)  # Форма массива — это кортеж из неотрицательных целых чисел,
# которые определяют количество элементов в каждом измерении.

arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
print(arr)  # Вывод массива
print(np.sort(arr))  # Возвращает отсортированную копию массива
print(np.argsort(arr))  # Возвращает индексы, по которым можно было бы отсортировать массив.
print(np.argmax(arr))  # Возвращает индекс максимального значения вдоль оси.
print(np.argmin(arr))  # Возвращает индексы минимальных значений вдоль оси.
print(np.mean(arr))  # среднее арифметическое
print(np.median(arr))  # медиану вдоль указанной оси.
print(np.std(arr))  # стандартное отклонение вдоль указанной оси.
print(np.var(arr))  # отклонение вдоль указанной оси.
print(np.sum(arr))  # Сумма элементов массива по заданной оси
print(np.cumsum(arr))  # Возвращает совокупную сумму элементов вдоль заданной оси
print(np.cumprod(arr))  # Возвращает суммарное произведение элементов вдоль заданной оси.

array_example = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],

                          [[0, 1, 2, 3],
                           [4, 5, 6, 7]],

                          [[0 ,1 ,2, 3],
                           [4, 5, 6, 7]]])
print(array_example.ndim)  # Размерность массива - количество измерений массива
print(array_example.size)  # Размер массива - количество элементов в массиве
print(array_example.shape)  # Форма массива - кортеж из неотрицательных целых чисел

# изменить форму массива
a = np.arange(6)
print(a)

b = a.reshape(3, 2)
print(b)

np.reshape(a, shape=(1, 6), order='C')
# https://numpy.org/doc/stable/user/absolute_beginners.html
