<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="logo.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# Выпускная квалификационная работа
# Проектирование и макетирование системы мониторинга БПЛА
</div>
<br>


Государственный университет "Дубна". Образовательное направление: 09.03.04 Программная инженерия

Автор:  Терешкин Дмитрий Александрович, 4253

Руководитель: доцент, Задорожный Александр Михайлович

# Содержание

- [Содержание](#содержание)
- [Краткое описание проекта](#краткое-описание-проекта)
  - [Сущность дипломного проекта](#сущность-дипломного-проекта)
  - [Научно-практическая значимость](#научно-практическая-значимость)
  - [Основные возможности системы](#основные-возможности-системы)
  - [Техническая архитектура](#техническая-архитектура)
- [Технологический стек проекта](#технологический-стек-проекта)
- [Структура проекта](#структура-проекта)
- [Архитектура и диаграммы проекта](#архитектура-и-диаграммы-проекта)
  - [Общая архитектура системы](#общая-архитектура-системы)
  - [Диаграмма классов](#диаграмма-классов)
  - [Диаграммы последовательности](#диаграммы-последовательности)
- [Начало работы](#начало-работы)
  - [Предварительные требования](#предварительные-требования)
  - [Установка](#установка)
  - [Использование](#использование)
- [Контакты](#контакты)

---
## Краткое описание проекта

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

# Архитектура и диаграммы проекта

## Общая архитектура системы

### Краткая архитектурная схема
![Краткая архитектура](диаграммы/архитектура-диплом-краткая.png)

### Подробная архитектурная схема  
![Подробная архитектура](диаграммы/архитектура-диплом-подробная.png)

## Диаграмма классов
![Диаграмма классов](диаграммы/диаграма_классов_диплом.png)

*Диаграмма классов демонстрирует основные компоненты системы: главный виджет приложения (Widget), модули конфигурации (Config), обработчики видео и измерения расстояний, а также специализированные модули для калибровки, синхронизации и измерения расстояний.*

## Диаграммы последовательности

### Общий процесс работы системы
![Общая диаграмма последовательности](диаграммы/диаграма-последовательности.png)

### Процесс детектирования объектов
![Детектирование объектов](диаграммы/диаграмма%20последовательности%20детекция.png)

### Процесс калибровки камер
![Калибровка камер](диаграммы/диаг%20посл%20калибровка.png)

*Процесс калибровки включает захват изображений шахматной доски, вычисление внутренних параметров камер и сохранение калибровочных данных.*

### Процесс синхронизации камер
![Синхронизация камер](диаграммы/диаг%20посл%20синхронизация.png)

*Синхронизация обеспечивает временное выравнивание кадров от двух камер для корректной работы стереозрения.*

### Процесс измерения расстояний
![Измерение расстояний](диаграммы/диаг%20посл%20измерение%20раст.png)

*Измерение расстояний основано на принципах стереозрения: анализе диспаратности между соответствующими точками на изображениях с двух камер.*

### Загрузка и применение настроек
![Настройки приложения](диаграммы/Диаграмма%20последовательности%20для%20загрузки%20и%20применения%20настроек.png)

*Система поддерживает гибкую конфигурацию через файл настроек, позволяя адаптировать параметры детекции и измерений под конкретные задачи.*

## Начало работы

### Предварительные требования

- **Язык программирования:** Python
- **Менеджер пакетов:** pip

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

## Контакты
ФИО: Терешкин Дмитрий Александрович

Email: alexrumling2000@gmail.com

TG: @Otrix_ai

VK: https://vk.com/otrix_ai

---
<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square

