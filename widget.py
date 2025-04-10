import sys
import cv2
import numpy as np
import os
import json
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QMessageBox, QFileDialog, QGroupBox, QRadioButton, QButtonGroup,
    QComboBox, QTabWidget, QStackedWidget, QScrollArea, QSizePolicy
)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon
from ultralytics import YOLO
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack
from camera_utils import VideoThread, convert_cv_qt
from settings_dialog import SettingsDialog
from config import Config
from calibration_module import CalibrationDialog
from sync_module import SyncDialog
from distance_module import DistanceCalculatorDialog, DistanceCalculationThread


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Мониторинг БПЛА")
        self.setMinimumSize(1000, 700)

        # Инициализация конфигурации
        self.config = Config()
        
        # Режимы работы приложения
        self.mode = "detection"  # default mode: detection, distance

        # Настройки модуля измерения расстояния (будут обновляться через диалог)
        self.distance_module_enabled = False
        self.distance_module_baseline = 10.0

        # Установка стилей для виджета
        self.setStyleSheet(
            """
            QWidget {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                              stop:0 #2C3E50, stop:1 #3498DB);
                font-family: Arial, sans-serif;
            }
            
            /* Стиль заголовка */
            QLabel#title {
                color: white;
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
                background-color: rgba(0, 0, 0, 0.2);
                border-radius: 10px;
                margin: 10px;
            }
            
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #2ECC71, stop:1 #27AE60);
                color: white;
                border: none;
                padding: 12px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin: 5px;
                transition-duration: 0.4s;
                cursor: pointer;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                min-height: 30px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #27AE60, stop:1 #229954);
                transform: translateY(-2px);
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
            }
            
            QPushButton:pressed {
                background-color: #229954;
                transform: translateY(1px);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            QLabel#video_label {
                background-color: black;
                border: 3px solid #2ECC71;
                border-radius: 15px;
                padding: 10px;
                margin: 10px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                min-width: 320px;
                min-height: 240px;
            }
            
            QTextEdit {
                background-color: rgba(255, 255, 255, 0.85);
                border: 2px solid #2ECC71;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
                font-size: 12px;
                color: #2C3E50;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                min-height: 100px;
            }
            
            QScrollBar:vertical {
                border: none;
                background-color: rgba(0, 0, 0, 0.1);
                width: 10px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #2ECC71;
                border-radius: 5px;
                min-height: 30px;
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            
            QGroupBox {
                background-color: rgba(0, 0, 0, 0.15);
                border: 1px solid #2ECC71;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
                color: white;
                font-weight: bold;
                min-height: 50px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: white;
                font-weight: bold;
            }
            
            QRadioButton {
                color: white;
                font-size: 14px;
                spacing: 8px;
                margin: 5px;
                color: #FFFFFF;
                font-size: 14px;
                background: transparent;
                padding: 2px;
            }
            
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            
            QComboBox {
                background-color: rgba(255, 255, 255, 0.8);
                border: 1px solid #2ECC71;
                border-radius: 5px;
                padding: 5px;
                min-width: 150px;
                color: #2C3E50;
            }
            
            QComboBox:hover {
                border: 1px solid #27AE60;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: #2ECC71;
                border-left-style: solid;
            }
            
            QScrollArea {
                border: none;
                background-color: transparent;
            }

            QLabel {
                color: #FFFFFF;
                font-size: 14px;
                background: transparent;
                padding: 2px;
            }

            """
        )

        # Добавляем заголовок
        self.title_label = QLabel("Система мониторинга БПЛА")
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignCenter)

        # Флаг успешного подключения камеры
        self.connected = False

        # Создаем основной layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)
        
        # Панель выбора режима работы
        self.mode_group = QGroupBox("Режим работы")
        mode_layout = QHBoxLayout()
        
        self.mode_detection = QRadioButton("Распознавание и трекинг")
        self.mode_detection.setChecked(True)
        self.mode_detection.toggled.connect(lambda checked: self.change_mode("detection") if checked else None)
        
        self.mode_distance = QRadioButton("Распознавание, трекинг и измерение расстояния")
        self.mode_distance.toggled.connect(lambda checked: self.change_mode("distance") if checked else None)
        
        mode_layout.addWidget(self.mode_detection)
        mode_layout.addWidget(self.mode_distance)
        mode_layout.addStretch()
        
        self.mode_group.setLayout(mode_layout)
        main_layout.addWidget(self.mode_group)
        
        # Создаем stacked widget для разных режимов
        self.stacked_widget = QStackedWidget()
        
        # Страница 1: Режим распознавания
        self.detection_widget = QWidget()
        detection_layout = QHBoxLayout()
        
        # Видео панель
        video_layout = QVBoxLayout()

        # Окно с видео
        self.video_label = QLabel("Ожидание видеопотока...")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(False)

        # Панель логов
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setMinimumHeight(100)
        self.log_text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        video_layout.addWidget(self.video_label, 7)
        video_layout.addWidget(self.log_text_edit, 3)

        # Панель управления (кнопки и список камер)
        control_layout = QVBoxLayout()
        
        # Кнопки управления
        buttons_group = QGroupBox("Управление")
        buttons_layout = QVBoxLayout()
        
        # Первый ряд кнопок
        main_buttons_layout = QHBoxLayout()
        
        self.model_button = QPushButton("📁 Выбор модели YOLO")
        self.model_button.clicked.connect(self.select_model)
        self.model_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.settings_button = QPushButton("⚙️ Настройки")
        self.settings_button.clicked.connect(self.show_settings)
        self.settings_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        main_buttons_layout.addWidget(self.model_button)
        main_buttons_layout.addWidget(self.settings_button)
        
        buttons_layout.addLayout(main_buttons_layout)
        buttons_group.setLayout(buttons_layout)
        control_layout.addWidget(buttons_group)
        
        # Контейнер для списка камер
        cameras_group = QGroupBox("Доступные источники видео")
        cameras_layout = QVBoxLayout()
        
        self.cameras_scroll = QScrollArea()
        self.cameras_scroll.setWidgetResizable(True)
        self.cameras_scroll.setMinimumHeight(150)
        
        self.cameras_widget = QWidget()
        self.cameras_container = QVBoxLayout(self.cameras_widget)
        self.cameras_container.setSpacing(5)
        self.cameras_container.setContentsMargins(5, 5, 5, 5)
        self.cameras_container.addStretch()
        
        self.cameras_scroll.setWidget(self.cameras_widget)
        cameras_layout.addWidget(self.cameras_scroll)
        
        cameras_group.setLayout(cameras_layout)
        control_layout.addWidget(cameras_group)
        control_layout.addStretch()
        
        detection_layout.addLayout(video_layout, 7)
        detection_layout.addLayout(control_layout, 3)
        
        self.detection_widget.setLayout(detection_layout)
        
        # Страница 2: Режим измерения расстояния
        self.distance_widget = QWidget()
        distance_layout = QHBoxLayout()
        
        # Видео панель (аналогично первой странице)
        distance_video_layout = QVBoxLayout()
        
        # Окно с видео
        self.distance_video_label = QLabel("Выберите две камеры для измерения расстояния")
        self.distance_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.distance_video_label.setMinimumSize(320, 240)
        self.distance_video_label.setObjectName("video_label")
        self.distance_video_label.setAlignment(Qt.AlignCenter)
        self.distance_video_label.setScaledContents(False)
        
        # Панель логов
        self.distance_log_text_edit = QTextEdit()
        self.distance_log_text_edit.setReadOnly(True)
        self.distance_log_text_edit.setMinimumHeight(100)
        self.distance_log_text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        distance_video_layout.addWidget(self.distance_video_label, 7)
        distance_video_layout.addWidget(self.distance_log_text_edit, 3)
        
        # Панель управления для режима измерения расстояния
        distance_control_layout = QVBoxLayout()
        
        # Настройки камер для измерения расстояния
        camera_selection_group = QGroupBox("Выбор камер для измерения")
        camera_selection_layout = QVBoxLayout()
        
        # Выбор первой камеры
        self.cam1_layout = QHBoxLayout()
        self.cam1_layout.addWidget(QLabel("Камера 1:"))
        self.cam1_combo = QComboBox()
        self.cam1_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cam1_layout.addWidget(self.cam1_combo)
        camera_selection_layout.addLayout(self.cam1_layout)
        
        # Выбор второй камеры
        self.cam2_layout = QHBoxLayout()
        self.cam2_layout.addWidget(QLabel("Камера 2:"))
        self.cam2_combo = QComboBox()
        self.cam2_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cam2_layout.addWidget(self.cam2_combo)
        camera_selection_layout.addLayout(self.cam2_layout)
        
        # Выбор активной камеры для отображения
        self.active_cam_layout = QHBoxLayout()
        self.active_cam_layout.addWidget(QLabel("Показывать:"))
        self.active_cam_combo = QComboBox()
        self.active_cam_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.active_cam_combo.addItem("Камера 1")
        self.active_cam_combo.addItem("Камера 2")
        self.active_cam_combo.currentIndexChanged.connect(self.on_camera_switch)
        self.active_cam_layout.addWidget(self.active_cam_combo)
        camera_selection_layout.addLayout(self.active_cam_layout)
        
        # Кнопки управления для режима расстояния
        measurement_buttons_layout = QHBoxLayout()
        self.start_distance_button = QPushButton("▶️ Запустить измерение")
        self.start_distance_button.clicked.connect(self.start_distance_measurement)
        self.start_distance_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.stop_distance_button = QPushButton("⏹️ Остановить")
        self.stop_distance_button.clicked.connect(self.stop_distance_measurement)
        self.stop_distance_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_distance_button.setEnabled(False)
        
        measurement_buttons_layout.addWidget(self.start_distance_button)
        measurement_buttons_layout.addWidget(self.stop_distance_button)
        
        camera_selection_layout.addLayout(measurement_buttons_layout)
        camera_selection_group.setLayout(camera_selection_layout)
        distance_control_layout.addWidget(camera_selection_group)
        
        # Группа статуса
        status_group = QGroupBox("Статус")
        status_layout = QVBoxLayout()
        
        self.calibration_status_label = QLabel("Калибровка: ❌")
        self.sync_status_label = QLabel("Синхронизация: ❌")
        
        status_layout.addWidget(self.calibration_status_label)
        status_layout.addWidget(self.sync_status_label)
        status_group.setLayout(status_layout)
        
        distance_control_layout.addWidget(status_group)
        
        # Кнопки калибровки и синхронизации
        tools_group = QGroupBox("Инструменты")
        tools_layout = QVBoxLayout()
        
        self.distance_calibration_button = QPushButton("🔍 Калибровка камер")
        self.distance_calibration_button.clicked.connect(self.open_calibration_dialog)
        self.distance_calibration_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.distance_sync_button = QPushButton("⏱️ Синхронизация камер")
        self.distance_sync_button.clicked.connect(self.open_sync_dialog)
        self.distance_sync_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Добавляем кнопку выбора модели
        self.distance_model_button = QPushButton("📁 Выбор модели YOLO")
        self.distance_model_button.clicked.connect(self.select_model)
        self.distance_model_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Добавляем кнопку настроек модели
        self.distance_settings_button = QPushButton("⚙️ Настройки")
        self.distance_settings_button.clicked.connect(self.show_settings)
        self.distance_settings_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        tools_layout.addWidget(self.distance_calibration_button)
        tools_layout.addWidget(self.distance_sync_button)
        tools_layout.addWidget(self.distance_model_button)
        tools_layout.addWidget(self.distance_settings_button)
        
        tools_group.setLayout(tools_layout)
        distance_control_layout.addWidget(tools_group)
        
        distance_control_layout.addStretch()
        
        distance_layout.addLayout(distance_video_layout, 7)
        distance_layout.addLayout(distance_control_layout, 3)
        
        self.distance_widget.setLayout(distance_layout)
        
        # Добавляем страницы в стековый виджет
        self.stacked_widget.addWidget(self.detection_widget)
        self.stacked_widget.addWidget(self.distance_widget)
        
        main_layout.addWidget(self.stacked_widget)
        
        self.setLayout(main_layout)

        # Переменные для работы с видеопотоком
        self.thread = None
        self.selected_camera_url = None
        self.model_path = self.config.get_last_model()

        # Загрузка камер из файла
        self.load_cameras()

        # Проверка статуса калибровки и синхронизации
        self.update_calibration_sync_status()

    def update_calibration_sync_status(self):
        # Обновляем статус калибровки
        calibration_status = self.config.get_calibration_status()
        if calibration_status.get('calibrated', False):
            self.calibration_status_label.setText("Калибровка: ✅")
        else:
            self.calibration_status_label.setText("Калибровка: ❌")
            
        # Обновляем статус синхронизации
        sync_status = self.config.get_sync_status()
        if sync_status.get('synced', False):
            self.sync_status_label.setText("Синхронизация: ✅")
        else:
            self.sync_status_label.setText("Синхронизация: ❌")

    def change_mode(self, mode):
        """Изменяет режим работы приложения."""
        if mode == self.mode:
            return
            
        # Останавливаем текущие потоки
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread = None
            
        self.mode = mode
        
        if mode == "detection":
            self.stacked_widget.setCurrentIndex(0)
            self.log_message("Режим распознавания и трекинга активирован")
        elif mode == "distance":
            self.stacked_widget.setCurrentIndex(1)
            self.distance_log_text_edit.append("Режим измерения расстояния активирован")
            
            # Проверяем калибровку и синхронизацию
            calibration_status = self.config.get_calibration_status()
            sync_status = self.config.get_sync_status()
            
            if not calibration_status.get('calibrated', False):
                self.distance_log_text_edit.append("⚠️ Камеры не откалиброваны! Рекомендуется выполнить калибровку.")
                
            if not sync_status.get('synced', False):
                self.distance_log_text_edit.append("⚠️ Камеры не синхронизированы! Рекомендуется выполнить синхронизацию.")

    def open_calibration_dialog(self):
        """Открывает диалог калибровки камер."""
        dialog = CalibrationDialog(self)
        if dialog.exec():
            # Обновляем статус после калибровки
            self.update_calibration_sync_status()
            self.log_message("Калибровка камер завершена", "green")

    def open_sync_dialog(self):
        """Открывает диалог синхронизации камер."""
        dialog = SyncDialog(self)
        if dialog.exec():
            # Обновляем статус после синхронизации
            self.update_calibration_sync_status()
            self.log_message("Синхронизация камер завершена", "green")

    def open_distance_calculator(self):
        """Открывает диалог расчета расстояния."""
        dialog = DistanceCalculatorDialog(self)
        dialog.exec()


    def start_distance_measurement(self):
        """Запускает процесс измерения расстояния непосредственно в основном интерфейсе."""
        # Убедимся, что камеры откалиброваны и синхронизированы
        calibration_status = self.config.get_calibration_status()
        sync_status = self.config.get_sync_status()
        
        # Получаем URL камер из данных комбобоксов
        camera1_url = self.cam1_combo.currentData()
        camera2_url = self.cam2_combo.currentData()
        
        if not camera1_url or not camera2_url:
            QMessageBox.warning(self, "Ошибка", "Выберите две разные камеры для измерения.")
            return
            
        if camera1_url == camera2_url:
            QMessageBox.warning(self, "Ошибка", "Выберите две разные камеры для измерения.")
            return
        
        # Проверяем, выбрана ли модель
        if not self.model_path:
            QMessageBox.warning(self, "Ошибка", "Необходимо выбрать модель YOLO. Нажмите кнопку 'Выбор модели YOLO'.")
            return
        
        warnings = []
        if not calibration_status.get('calibrated', False):
            warnings.append("Камеры не откалиброваны")
            
        if not sync_status.get('synced', False):
            warnings.append("Камеры не синхронизированы")
            
        if warnings:
            warning_text = "Обнаружены проблемы:\n" + "\n".join([f"- {w}" for w in warnings])
            warning_text += "\n\nИзмерение расстояния может быть неточным. Продолжить?"
            
            result = QMessageBox.warning(
                self, 
                "Предупреждение", 
                warning_text,
                QMessageBox.Yes | QMessageBox.No
            )
            
            if result == QMessageBox.No:
                return
        
        # Останавливаем текущий поток распознавания, если он запущен
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread = None
        
        # Если уже есть активный поток измерения расстояния, остановим его
        if hasattr(self, 'distance_thread') and self.distance_thread and self.distance_thread.isRunning():
            self.distance_thread.stop()
            
        # Загружаем данные калибровки и синхронизации
        calibration_data = {}
        sync_data = {}
        
        # Загружаем файл калибровки
        if os.path.exists("calibration_data.json"):
            try:
                with open("calibration_data.json", "r") as f:
                    calibration_data = json.load(f)
            except Exception as e:
                self.log_message(f"Ошибка при загрузке данных калибровки: {e}", "red")
        
        # Загружаем файл синхронизации
        if os.path.exists("sync_data.json"):
            try:
                with open("sync_data.json", "r") as f:
                    sync_data = json.load(f)
            except Exception as e:
                self.log_message(f"Ошибка при загрузке данных синхронизации: {e}", "red")
        
        # Получаем значение базиса из настроек
        baseline = 10.0  # Значение по умолчанию
        distance_settings = self.config.get_distance_measure_settings()
        if 'baseline' in distance_settings:
            baseline = distance_settings['baseline']
        
        # Создаем новый поток для измерения расстояния
        self.distance_thread = DistanceCalculationThread(
            camera1_url, camera2_url, self.model_path, baseline, 
            calibration_data, sync_data
        )
        
        # Подключаем сигналы
        self.distance_thread.frame_signal.connect(self.update_distance_frame)
        self.distance_thread.error_signal.connect(self.on_distance_error)
        
        # Запускаем поток
        self.distance_thread.start()
        
        # Обновляем UI
        self.distance_log_text_edit.clear()
        self.distance_log_text_edit.append(f"Измерение расстояния запущено")
        self.distance_log_text_edit.append(f"Камера 1: {self.cam1_combo.currentText()} ({camera1_url})")
        self.distance_log_text_edit.append(f"Камера 2: {self.cam2_combo.currentText()} ({camera2_url})")
        self.distance_log_text_edit.append(f"Модель: {self.model_path}")
        self.distance_log_text_edit.append(f"Базис: {baseline} см")
        
        # Отключаем кнопку запуска и включаем кнопку остановки
        self.start_distance_button.setEnabled(False)
        self.stop_distance_button.setEnabled(True)
    
    def update_distance_frame(self, original_frame, processed_frame, info):
        """Обновляет кадр и информацию о расстоянии в интерфейсе."""
        if original_frame is None or processed_frame is None:
            return  # Пропускаем обработку, если один из кадров отсутствует
            
        # Определяем, какую камеру показывать
        display_index = self.active_cam_combo.currentIndex()
        
        # Добавляем отладочную информацию один раз в 100 кадров
        if not hasattr(self, 'debug_counter'):
            self.debug_counter = 0
        
        self.debug_counter += 1
        if self.debug_counter % 100 == 0:
            # Очищаем старые отладочные сообщения
            self.distance_log_text_edit.clear()
            self.distance_log_text_edit.append(f"Текущий индекс камеры: {display_index}")
            self.distance_log_text_edit.append(f"Камера 1: {self.cam1_combo.currentText()}")
            self.distance_log_text_edit.append(f"Камера 2: {self.cam2_combo.currentText()}")
        
        # Выбираем кадр на основе индекса активной камеры
        # Индекс 0 - Камера 1 (original_frame), Индекс 1 - Камера 2 (processed_frame)
        frame_to_display = original_frame if display_index == 0 else processed_frame
        
        # Сохраняем последний кадр для возможного ресайза
        self.last_distance_frame = frame_to_display.copy()
        
        # Также сохраняем оба кадра отдельно для возможности переключения
        self.cam1_frame = original_frame.copy()
        self.cam2_frame = processed_frame.copy()
        
        # Конвертируем кадр для отображения в формат Qt
        qt_img = convert_cv_qt(frame_to_display)
        
        # Создаем QPixmap из QImage
        pixmap = QPixmap.fromImage(qt_img)
        
        # Масштабируем изображение с сохранением пропорций
        scaled_pixmap = pixmap.scaled(
            self.distance_video_label.width(), 
            self.distance_video_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # Отображаем на метке
        self.distance_video_label.setPixmap(scaled_pixmap)
        self.distance_video_label.setAlignment(Qt.AlignCenter)
        
        # Обновляем информацию о распознанных объектах в лог
        detections = info.get('detections', {})
        if detections and self.debug_counter % 30 == 0:  # Обновляем каждые 30 кадров
            self.distance_log_text_edit.append("Распознанные объекты:")
            
            # Добавляем только первые 5 объектов для экономии места в логе
            for i, (obj_id, obj_data) in enumerate(list(detections.items())[:5]):
                cls_name = obj_data['class']
                distance = obj_data['distance']
                confidence = obj_data['confidence']
                
                # Определяем цвет в зависимости от расстояния
                if distance < 2:  # ближе 2 метров
                    color = "red"
                elif distance < 5:  # 2-5 метров
                    color = "orange"
                else:
                    color = "green"
                
                # Добавляем сообщение с форматированием
                self.distance_log_text_edit.append(
                    f'<span style="color: {color};"><b>{cls_name}</b> (ID: {obj_id.split("_")[1]}): '
                    f'{distance:.2f} м (увер.: {confidence:.2f})</span>'
                )
                
    def on_distance_error(self, error_msg):
        """Обрабатывает ошибки в потоке измерения расстояния."""
        QMessageBox.critical(self, "Ошибка", error_msg)
        self.stop_distance_measurement()

    def stop_distance_measurement(self):
        """Останавливает процесс измерения расстояния."""
        if hasattr(self, 'distance_thread') and self.distance_thread and self.distance_thread.isRunning():
            self.distance_thread.stop()
            self.distance_thread = None
            
        # Очищаем изображение
        self.distance_video_label.setText("Выберите две камеры для измерения расстояния")
        self.distance_video_label.setPixmap(QPixmap())
        
        self.distance_log_text_edit.append("Измерение расстояния остановлено")
        self.start_distance_button.setEnabled(True)
        self.stop_distance_button.setEnabled(False)

    def show_settings(self):
        """Открывает диалог настроек."""
        dialog = SettingsDialog(self, self.model_path)
        
        # Загружаем текущие настройки
        detection_settings = self.config.get_detection_settings()
        settings = {
            'conf': detection_settings['confidence_threshold'],
            'iou': detection_settings['iou_threshold'],
            'device': 'cpu',  # Значение по умолчанию
            'half': False,    # Значение по умолчанию
            'fps': 30        # Значение по умолчанию для FPS
        }
        dialog.set_settings(settings)
        
        if dialog.exec():
            # Получаем и сохраняем новые настройки
            new_settings = dialog.get_settings()
            
            # Обновляем настройки детекции
            self.config.set_detection_settings(
                confidence_threshold=new_settings.get('conf'),
                iou_threshold=new_settings.get('iou')
            )
            
            # Если поток запущен, применяем новые настройки
            if self.thread and self.thread.isRunning():
                self.thread.update_settings(new_settings)
                self.log_message("Настройки успешно обновлены", "green", both_logs=True)
            else:
                self.log_message("Настройки сохранены и будут применены при запуске видеопотока", "blue", both_logs=True)

    def log_message(self, message, color="black", both_logs=False):
        """Добавляет сообщение в лог с цветовым форматированием.
        
        Args:
            message: Текст сообщения
            color: Цвет текста
            both_logs: Если True, отображает сообщение в обоих логах, независимо от текущего режима
        """
        formatted_message = f'<span style="color: {color};">{message}</span>'
        
        if both_logs or self.mode == "detection":
            self.log_text_edit.append(formatted_message)
            
        if both_logs or self.mode == "distance":
            self.distance_log_text_edit.append(formatted_message)

    def load_cameras(self):
        """Загружает список камер из файла cameras.txt и создаёт кнопки для них."""
        try:
            with open("cameras.txt", "r") as f:
                camera_lines = [line.strip() for line in f if line.strip()]
            if not camera_lines:
                QMessageBox.warning(self, "Ошибка", "Файл cameras.txt пуст.")
                return
                
            # Очищаем существующие списки
            # Удаляем все виджеты из контейнера
            while self.cameras_container.count():
                item = self.cameras_container.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                    
            self.cam1_combo.clear()
            self.cam2_combo.clear()
            
            # Обработка строк из cameras.txt
            cameras = []
            camera_names = []
            
            for line in camera_lines:
                parts = line.strip().split(' ', 1)  # Разделяем на имя и URL
                if len(parts) == 2:
                    camera_name = parts[0]
                    camera_url = parts[1]
                    cameras.append(camera_url)
                    camera_names.append(camera_name)
                else:
                    # Если формат неверный, используем всю строку как URL
                    cameras.append(line)
                    camera_names.append(f"Камера {len(cameras)}")
                
            # Добавляем камеры в основной список и выпадающие списки
            self.log_message("Камеры успешно загружены.", "black")
            
            # Используем градиентные цвета для кнопок
            button_colors = [
                "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9)",
                "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2ecc71, stop:1 #27ae60)",
                "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b)",
                "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #9b59b6, stop:1 #8e44ad)",
                "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f1c40f, stop:1 #f39c12)"
            ]
            
            for i, (camera_url, camera_name) in enumerate(zip(cameras, camera_names)):
                # Кнопки для режима распознавания с иконкой камеры и именем из файла
                btn = QPushButton(f"📹 {camera_name}")
                btn.clicked.connect(lambda ch, url=camera_url: self.select_camera(url))
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                btn.setMinimumHeight(40)
                
                # Устанавливаем разные цвета для кнопок по циклу
                color_index = i % len(button_colors)
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {button_colors[color_index]};
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                            stop:0 #3aaef0, stop:1 #2980b9);
                    }}
                """)
                
                self.cameras_container.addWidget(btn)
                
                # Добавляем в выпадающие списки для режима расстояния
                self.cam1_combo.addItem(f"{camera_name}", camera_url)
                self.cam2_combo.addItem(f"{camera_name}", camera_url)
                
            # Добавляем растягивающийся элемент в конец списка камер
            self.cameras_container.addStretch()
                
            # Устанавливаем разные камеры по умолчанию для режима расстояния
            if len(cameras) > 1:
                self.cam2_combo.setCurrentIndex(1)
                
            # Получаем последние настройки дистанции из конфигурации
            if hasattr(self.config, 'get_distance_measure_settings'):
                distance_settings = self.config.get_distance_measure_settings()
                if 'cameras' in distance_settings and len(distance_settings['cameras']) >= 2:
                    # Устанавливаем сохраненные камеры
                    cam1 = distance_settings['cameras'][0]
                    cam2 = distance_settings['cameras'][1]
                    
                    # Находим сохраненные камеры по URL в данных комбобоксов
                    for i in range(self.cam1_combo.count()):
                        if self.cam1_combo.itemData(i) == cam1:
                            self.cam1_combo.setCurrentIndex(i)
                            break
                            
                    for i in range(self.cam2_combo.count()):
                        if self.cam2_combo.itemData(i) == cam2:
                            self.cam2_combo.setCurrentIndex(i)
                            break
                    
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Файл cameras.txt не найден. Пожалуйста, создайте файл со списком камер."
            )
            sys.exit(1)

    @Slot()
    def select_model(self):
        """Выбор модели YOLO (.pt файл)."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите модель YOLO",
            "",
            "YOLO Model (*.pt)"
        )
        if file_name:
            self.model_path = file_name
            self.config.set_last_model(file_name)
            self.log_message(f"Выбрана модель: {file_name}", "blue", both_logs=True)
            
            # Если поток уже запущен, обновляем модель
            if self.thread and self.thread.isRunning():
                if self.thread.set_model(self.model_path):
                    self.log_message("Модель успешно загружена", "green", both_logs=True)
                else:
                    self.log_message("Ошибка при загрузке модели", "red", both_logs=True)

    @Slot()
    def select_camera(self, camera_url):
        """Выбор и запуск видеопотока для выбранной камеры с предварительной проверкой доступности."""
        self.log_message(f"Выбрана камера: {camera_url}", "black")
        self.selected_camera_url = camera_url
        self.connected = False
        
        # Если предыдущий поток активен, остановим его
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread = None
        
        # Проверка доступности камеры
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            self.log_message(f"Камера {camera_url} недоступна.", "red")
            cap.release()
            return
        cap.release()
        
        # Получаем текущие настройки
        detection_settings = self.config.get_detection_settings()
        
        # Запуск нового потока с настройками
        self.thread = VideoThread(
            camera_url,
            conf=detection_settings['confidence_threshold'],
            iou=detection_settings['iou_threshold'],
            device='cpu',  # Значение по умолчанию
            half=False,    # Значение по умолчанию для half-precision
            fps=30         # Значение по умолчанию для FPS
        )
        self.thread.change_pixmap_signal.connect(self.update_video_frame)
        self.thread.detection_signal.connect(self.log_message)
        
        # Если модель уже была выбрана, загружаем её
        if self.model_path:
            if self.thread.set_model(self.model_path):
                self.log_message("Модель успешно загружена", "green")
            else:
                self.log_message("Ошибка при загрузке модели", "red")
        
        self.thread.start()
        
        # Сохраняем выбранную камеру в конфигурации
        self.config.set_last_camera(camera_url)

    @Slot(object)
    def update_video_frame(self, frame):
        """Обновляет кадр в окне видео."""
        if frame is not None:
            # Сохраняем последний кадр для возможного ресайза
            self.last_frame = frame
            
            # Конвертируем в QImage
            qt_img = convert_cv_qt(frame)
            
            # Создаем QPixmap из QImage
            pixmap = QPixmap.fromImage(qt_img)
            
            # Определяем какой label обновлять в зависимости от режима
            target_label = self.video_label if self.mode == "detection" else self.distance_video_label
            
            # Масштабируем изображение с сохранением пропорций
            scaled_pixmap = pixmap.scaled(
                target_label.width(), 
                target_label.height(),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            # Обновляем изображение
            target_label.setPixmap(scaled_pixmap)
            target_label.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        """Обработчик события изменения размера окна."""
        super().resizeEvent(event)
        
        # Обновляем видео в режиме распознавания, если есть кадр
        if hasattr(self, 'last_frame') and self.last_frame is not None:
            # В режиме распознавания используем update_video_frame
            if self.mode == "detection":
                self.update_video_frame(self.last_frame)
            # В режиме расстояния обновляем масштабирование напрямую
            elif self.mode == "distance" and self.distance_video_label.pixmap():
                qt_img = convert_cv_qt(self.last_frame)
                pixmap = QPixmap.fromImage(qt_img)
                scaled_pixmap = pixmap.scaled(
                    self.distance_video_label.width(), 
                    self.distance_video_label.height(),
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.distance_video_label.setPixmap(scaled_pixmap)
                self.distance_video_label.setAlignment(Qt.AlignCenter)

    def closeEvent(self, event):
        # Останавливаем основной поток распознавания
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            
        # Останавливаем поток измерения расстояния, если он запущен
        if hasattr(self, 'distance_thread') and self.distance_thread and self.distance_thread.isRunning():
            self.distance_thread.stop()
            
        event.accept()

    def on_camera_switch(self, index):
        """Обработчик переключения камеры в выпадающем списке."""
        self.distance_log_text_edit.append(f"Переключение на камеру: {index + 1}")
        
        # Если у нас есть сохраненные кадры, сразу переключаем отображение
        if hasattr(self, 'cam1_frame') and hasattr(self, 'cam2_frame'):
            if index == 0 and self.cam1_frame is not None:
                self.update_camera_display(self.cam1_frame)
                self.distance_log_text_edit.append("Отображение переключено на камеру 1")
            elif index == 1 and self.cam2_frame is not None:
                self.update_camera_display(self.cam2_frame)
                self.distance_log_text_edit.append("Отображение переключено на камеру 2")
    
    def update_camera_display(self, frame):
        """Обновляет отображение выбранной камеры."""
        if frame is None:
            return
            
        # Конвертируем кадр для отображения в формат Qt
        qt_img = convert_cv_qt(frame)
        
        # Создаем QPixmap из QImage
        pixmap = QPixmap.fromImage(qt_img)
        
        # Масштабируем изображение с сохранением пропорций
        scaled_pixmap = pixmap.scaled(
            self.distance_video_label.width(), 
            self.distance_video_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # Отображаем на метке
        self.distance_video_label.setPixmap(scaled_pixmap)
        self.distance_video_label.setAlignment(Qt.AlignCenter)
        
    def refresh_video_stream(self):
        """Обновляет текущий поток видео."""
        if hasattr(self, 'distance_thread') and self.distance_thread and self.distance_thread.isRunning():
            # Получаем текущий индекс выбранной камеры
            current_camera_index = self.active_cam_combo.currentIndex()
            self.distance_log_text_edit.append(f"Принудительное обновление потока для камеры {current_camera_index + 1}")
            
            # Если у нас есть сохраненные кадры, сразу переключаем отображение
            if current_camera_index == 0 and hasattr(self, 'cam1_frame') and self.cam1_frame is not None:
                self.update_camera_display(self.cam1_frame)
                self.distance_log_text_edit.append("Переключено на камеру 1")
            elif current_camera_index == 1 and hasattr(self, 'cam2_frame') and self.cam2_frame is not None:
                self.update_camera_display(self.cam2_frame)
                self.distance_log_text_edit.append("Переключено на камеру 2")
            else:
                self.distance_log_text_edit.append("Сохраненные кадры не найдены, перезапуск потока...")
                # Перезапускаем поток, если нет сохраненных кадров
                self.stop_distance_measurement()
                self.start_distance_measurement()

# Точка входа в приложение
if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
