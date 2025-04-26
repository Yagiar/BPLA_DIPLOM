<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="logo.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# BPLA_DIPLOM

<!-- default option, no dependency badges. -->


<!-- default option, no dependency badges. -->

</div>
<br>



## Содержание

- [Содержание](#содержание)
- [Структура проекта](#структура-проекта)
- [Начало работы](#начало-работы)
    - [Предварительные требования](#предварительные-требования)
    - [Установка](#установка)
    - [Использование](#использование)

---

## Структура проекта

```sh
└── BPLA_DIPLOM/
    ├── models/                      # Каталог с моделями YOLO
    │   ├── AOD_detection_model_yolo12n.pt
    │   ├── BPLA_model_10-11-2024.pt
    │   ├── united_datasets_airplane_birds_drone_11-03-2025.pt
    │   └── yolov8m_coco.pt
    ├── src/                         # Исходный код проекта
    │   ├── core/                    # Ядро приложения
    │   │   ├── __init__.py
    │   │   ├── config.py            # Управление конфигурацией
    │   │   └── distance_logic.py    # Логика для измерения расстояний
    │   ├── handlers/                # Обработчики событий и потоков
    │   │   ├── __init__.py
    │   │   ├── distance_handler.py  # Обработчик измерения расстояний
    │   │   ├── log_manager.py       # Управление логами
    │   │   └── video_handler.py     # Обработчик видеопотока
    │   ├── modules/                 # Функциональные модули
    │   │   ├── __init__.py
    │   │   ├── calibration_module.py # Модуль калибровки камер
    │   │   ├── distance_module.py   # Модуль измерения расстояний
    │   │   └── sync_module.py       # Модуль синхронизации камер
    │   ├── ui/                      # Компоненты пользовательского интерфейса
    │   │   ├── __init__.py
    │   │   ├── app_styles.py        # Стили приложения
    │   │   ├── settings_dialog.py   # Диалог настроек
    │   │   └── ui_components.py     # Фабрика UI компонентов
    │   ├── utils/                   # Вспомогательные утилиты
    │   │   ├── __init__.py
    │   │   ├── camera_loader.py     # Загрузка камер из файла
    │   │   └── camera_utils.py      # Утилиты для работы с камерами
    │   ├── __init__.py
    │   └── widget.py                # Основной виджет приложения
    ├── videos/                      # Тестовые видеофайлы
    │   ├── airplane.mp4
    │   ├── birds.mp4
    │   ├── kakoi-to-drone-vodyanoy.mp4
    │   ├── roi_dronov_vodyanoy.mp4
    │   └── rutube-uav-fire.mp4
    ├── .gitignore
    ├── cameras.txt                  # Список камер/видеопотоков
    ├── LICENSE
    ├── logo.png                     # Логотип приложения
    ├── main.py                      # Точка входа в приложение
    ├── README.md
    ├── requirements.txt             # Зависимости проекта
    └── settings.json                # Файл конфигурации
```

## Начало работы

### Предварительные требования

Этот проект требует следующих зависимостей:

- **Язык программирования:** Python
- **Менеджер пакетов:** Pip

### Установка

Соберите BPLA_DIPLOM из исходного кода и установите зависимости:

1. **Клонируйте репозиторий:**

    ```sh
      git clone https://github.com/Yagiar/BPLA_DIPLOM
    ```

2. **Перейдите в директорию проекта:**

    ```sh
      cd BPLA_DIPLOM
    ```

3. **Установите зависимости:**

	```sh
	  pip install -r requirements.txt
	```

### Использование

Запустите проект с помощью:


```sh
   python main.py
```
---

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square

