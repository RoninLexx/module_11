import inspect


def introspection_info(obj):  # Проводит интроспекцию объекта и возвращает словарь с его информацией.


    # Создаем словарь для хранения информации об объекте
    info = {"type": type(obj).__name__, "attributes": [], "methods": [],}

    # Получение атрибутов
    for attr in dir(obj):
        if not attr.startswith("__"):  # Игнорируем встроенные атрибуты
            info["attributes"].append(attr)

    # Получение методов
    for method_name, method in inspect.getmembers(obj, inspect.ismethod):
        info["methods"].append(method_name)

    # Получение других свойств в зависимости от типа объекта
    if isinstance(obj, int):
        info["bit_length"] = obj.bit_length()
    elif isinstance(obj, str):
        info["length"] = len(obj)
        info["is_upper"] = obj.isupper()
    elif isinstance(obj, list):
        info["length"] = len(obj)
        info["element_types"] = list(set(type(elem).__name__ for elem in obj))  # Отличные типы элементов из списка

    return info


if __name__ == "__main__":
    # Пример с числом
    number_info = introspection_info(42)
    print(number_info)


    # Пример с классом
    class MyClass:
        def __init__(self, name):
            self.name = name

        def greet(self):
            return f"Hello, {self.name}!"


    my_object = MyClass("Lexx")
    my_class_info = introspection_info(my_object)
    print(my_class_info)

    # Пример со строкой
    string_info = introspection_info("Hello, World!")
    print(string_info)

    # Пример со списком
    list_info = introspection_info([1, 2, 3, "text", 3.14])
    print(list_info)
