import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QMessageBox, QFileDialog
)
from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtGui import QImage, QPixmap
from ultralytics import YOLO
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack
from camera_utils import VideoThread, convert_cv_qt
from settings_dialog import SettingsDialog
from config import Config
from distance_measure_dialog import DistanceMeasureDialog


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Мониторинг БПЛА")
        self.setMinimumSize(1730, 970)

        # Инициализация конфигурации
        self.config = Config()

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
                padding: 15px 25px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 8px 4px;
                transition-duration: 0.4s;
                cursor: pointer;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
            }
            
            QTextEdit {
                background-color: rgba(255, 255, 255, 0.9);
                border: 2px solid #2ECC71;
                border-radius: 15px;
                padding: 15px;
                margin: 10px;
                font-size: 14px;
                color: #2C3E50;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
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
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

        # Добавляем заголовок
        self.title_label = QLabel("Система мониторинга БПЛА")
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignCenter)

        # Флаг успешного подключения камеры
        self.connected = False

        # Кнопки управления
        self.model_button = QPushButton("📁 Выбор модели YOLO")
        self.model_button.clicked.connect(self.select_model)
        
        self.settings_button = QPushButton("⚙️ Настройки")
        self.settings_button.clicked.connect(self.show_settings)
        
        self.distance_button = QPushButton("📏 Измерение расстояния")
        self.distance_button.clicked.connect(self.open_distance_measure_dialog)

        # Контейнер для списка камер
        self.cameras_container = QVBoxLayout()
        self.cameras_container.setSpacing(10)
        self.cameras_container.setContentsMargins(10, 10, 10, 10)

        # Создадим виджет-обёртку для кнопок камер
        self.cameras_widget = QWidget()
        self.cameras_widget.setLayout(self.cameras_container)

        # Окно с видео
        self.video_label = QLabel("Ожидание видеопотока...")
        self.video_label.setFixedSize(800, 600)
        self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignCenter)

        # Панель логов
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setMinimumHeight(200)

        # Компоновка интерфейса
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)
        
        content_layout = QHBoxLayout()
        
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.log_text_edit)
        
        control_layout = QVBoxLayout()
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.model_button)
        buttons_layout.addWidget(self.settings_button)
        buttons_layout.addWidget(self.distance_button)
        control_layout.addLayout(buttons_layout)
        control_layout.addWidget(self.cameras_widget)
        control_layout.addStretch()
        
        content_layout.addLayout(video_layout, stretch=7)
        content_layout.addLayout(control_layout, stretch=3)
        
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

        # Переменные для работы с видеопотоком
        self.thread = None
        self.selected_camera_url = None
        self.model_path = self.config.get_model_settings()['path']

        # Загрузка камер из файла
        self.load_cameras()

    def open_distance_measure_dialog(self):
        """Открывает диалог измерения расстояния."""
        dialog = DistanceMeasureDialog(self)
        if dialog.exec():
            enabled, selected_cams, baseline = dialog.get_values()
            self.distance_module_enabled = enabled
            self.distance_module_baseline = baseline
            if not enabled:
                self.log_message("Модуль измерения расстояния отключен.", "blue")
            else:
                self.log_message(f"Выбраны камеры для измерения: {selected_cams}, базис: {baseline}", "blue")

    def show_settings(self):
        """Открывает диалог настроек."""
        dialog = SettingsDialog(self, self.model_path)
        
        # Загружаем текущие настройки
        model_settings = self.config.get_model_settings()
        tracker_settings = self.config.get_tracker_settings()
        settings = {
            'conf': model_settings['conf'],
            'iou': model_settings['iou'],
            'device': model_settings['device'],
            'half': model_settings['half'],
            'fps': tracker_settings['fps']
        }
        dialog.set_settings(settings)
        
        if dialog.exec():
            # Получаем и сохраняем новые настройки
            new_settings = dialog.get_settings()
            self.config.update_settings(new_settings)
            
            # Если поток запущен, применяем новые настройки
            if self.thread and self.thread.isRunning():
                self.thread.update_settings(new_settings)
                self.log_message("Настройки успешно обновлены", "green")

    def log_message(self, message, color="black"):
        """Добавляет сообщение в лог с цветовым форматированием."""
        self.log_text_edit.append(f'<span style="color: {color};">{message}</span>')

    def load_cameras(self):
        """Загружает список камер из файла cameras.txt и создаёт кнопки для них."""
        try:
            with open("cameras.txt", "r") as f:
                cameras = [line.strip() for line in f if line.strip()]
            if not cameras:
                QMessageBox.warning(self, "Ошибка", "Файл cameras.txt пуст.")
                return
            self.log_message("Камеры успешно загружены.", "black")
            for camera_url in cameras:
                btn = QPushButton(camera_url)
                btn.clicked.connect(lambda ch, url=camera_url: self.select_camera(url))
                self.cameras_container.addWidget(btn)
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
            self.config.update_model_path(file_name)
            self.log_message(f"Выбрана модель: {file_name}", "blue")
            
            # Если поток уже запущен, обновляем модель
            if self.thread and self.thread.isRunning():
                if self.thread.set_model(self.model_path):
                    self.log_message("Модель успешно загружена", "green")
                else:
                    self.log_message("Ошибка при загрузке модели", "red")

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
        model_settings = self.config.get_model_settings()
        tracker_settings = self.config.get_tracker_settings()
        
        # Запуск нового потока с настройками
        self.thread = VideoThread(
            camera_url,
            conf=model_settings['conf'],
            iou=model_settings['iou'],
            device=model_settings['device'],
            half=model_settings['half'],
            fps=tracker_settings['fps']
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

    @Slot(object)
    def update_video_frame(self, frame):
        """Обновляет кадр в окне видео."""
        if frame is not None:
            self.video_label.setPixmap(QPixmap.fromImage(convert_cv_qt(frame)))

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
