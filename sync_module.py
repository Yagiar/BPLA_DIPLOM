import sys
import cv2
import numpy as np
import time
import json
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QComboBox, QGridLayout, QSpinBox, QDoubleSpinBox, QMessageBox,
    QGroupBox, QApplication, QFileDialog, QCheckBox, QProgressBar
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from camera_utils import convert_cv_qt
from config import Config


class FlashDetector:
    """
    Класс для обнаружения вспышек в кадрах видео.
    Используется для синхронизации камер по опорной точке.
    """
    def __init__(self, threshold=220, min_pixels=1000):
        self.threshold = threshold  # порог яркости для вспышки
        self.min_pixels = min_pixels  # минимальное количество ярких пикселей для детекции вспышки
        self.detected_flash = False
        self.flash_time = 0
        
    def detect(self, frame):
        """
        Обнаруживает вспышку в кадре.
        Возвращает True, если вспышка обнаружена в текущем кадре, иначе False.
        """
        # Преобразование в grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Подсчет пикселей, превышающих порог яркости
        bright_pixels = cv2.countNonZero(cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)[1])
        
        # Если количество ярких пикселей превышает минимум и вспышка ещё не обнаружена, 
        # считаем это вспышкой
        if bright_pixels > self.min_pixels and not self.detected_flash:
            self.detected_flash = True
            self.flash_time = time.time()
            return True
        
        # Если яркость упала ниже порога, сбрасываем состояние детектора
        if bright_pixels < self.min_pixels:
            self.detected_flash = False
            
        return False


class SyncThread(QThread):
    """
    Поток для синхронизации камер с использованием внешней вспышки.
    """
    update_signal = Signal(object, str)  # frame, camera_name
    status_signal = Signal(str)  # status message
    finished_signal = Signal(dict)  # synchronization results
    
    def __init__(self, camera1_url, camera2_url, flash_threshold=220, min_pixels=1000, max_wait_time=30):
        super().__init__()
        self.camera1_url = camera1_url
        self.camera2_url = camera2_url
        self.flash_threshold = flash_threshold
        self.min_pixels = min_pixels
        self.max_wait_time = max_wait_time  # максимальное время ожидания вспышки в секундах
        self.running = False
        
    def stop(self):
        self.running = False
        
    def run(self):
        self.running = True
        self.status_signal.emit("Инициализация камер...")
        
        # Открытие камер
        cap1 = cv2.VideoCapture(self.camera1_url)
        if not cap1.isOpened():
            self.status_signal.emit(f"Ошибка: Не удалось открыть камеру 1 ({self.camera1_url})")
            return
        
        cap2 = cv2.VideoCapture(self.camera2_url)
        if not cap2.isOpened():
            cap1.release()
            self.status_signal.emit(f"Ошибка: Не удалось открыть камеру 2 ({self.camera2_url})")
            return
        
        # Инициализация детекторов вспышек
        detector1 = FlashDetector(self.flash_threshold, self.min_pixels)
        detector2 = FlashDetector(self.flash_threshold, self.min_pixels)
        
        flash1_detected = False
        flash2_detected = False
        
        flash1_time = 0
        flash2_time = 0
        
        start_time = time.time()
        
        self.status_signal.emit(
            "Ожидание вспышки для синхронизации...\n"
            "Используйте фотовспышку или быстро включите/выключите свет для синхронизации."
        )
        
        # Основной цикл обработки
        while self.running and (not flash1_detected or not flash2_detected):
            # Проверка таймаута
            if time.time() - start_time > self.max_wait_time:
                self.status_signal.emit("Превышено время ожидания. Синхронизация прервана.")
                cap1.release()
                cap2.release()
                return
            
            # Чтение кадров с камер
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                self.status_signal.emit("Ошибка: Не удалось получить кадр с одной из камер")
                break
            
            # Обнаружение вспышки в первой камере
            if not flash1_detected and detector1.detect(frame1):
                flash1_detected = True
                flash1_time = detector1.flash_time
                self.status_signal.emit(f"Вспышка обнаружена на камере 1 в {flash1_time:.6f}")
            
            # Обнаружение вспышки во второй камере
            if not flash2_detected and detector2.detect(frame2):
                flash2_detected = True
                flash2_time = detector2.flash_time
                self.status_signal.emit(f"Вспышка обнаружена на камере 2 в {flash2_time:.6f}")
            
            # Отображение кадров
            self.update_signal.emit(frame1, "camera1")
            self.update_signal.emit(frame2, "camera2")
            
            # Небольшая задержка
            time.sleep(0.01)
        
        # Освобождение камер
        cap1.release()
        cap2.release()
        
        if not self.running:
            self.status_signal.emit("Синхронизация прервана")
            return
        
        # Вычисление разницы во времени между камерами
        if flash1_detected and flash2_detected:
            time_diff = flash2_time - flash1_time
            self.status_signal.emit(f"Синхронизация завершена. Разница во времени: {time_diff:.6f} секунд")
            
            # Подготовка результатов
            sync_data = {
                "camera1": {
                    "url": self.camera1_url,
                    "flash_time": flash1_time
                },
                "camera2": {
                    "url": self.camera2_url,
                    "flash_time": flash2_time
                },
                "time_diff": time_diff,
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.finished_signal.emit(sync_data)
        else:
            self.status_signal.emit("Не удалось обнаружить вспышку на обеих камерах")


class SyncDialog(QDialog):
    """
    Диалоговое окно для синхронизации камер.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.config = Config()
        
        self.setWindowTitle("Синхронизация камер")
        self.resize(1200, 800)
        
        # Инициализация интерфейса
        self.init_ui()
        
        # Загрузка камер
        self.load_cameras()
        
        # Переменные для синхронизации
        self.sync_thread = None
        self.sync_data = None
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        main_layout = QVBoxLayout()
        
        # Выбор камер
        cameras_group = QGroupBox("Выбор камер")
        cameras_layout = QGridLayout()
        
        cameras_layout.addWidget(QLabel("Камера 1:"), 0, 0)
        self.camera1_combo = QComboBox()
        cameras_layout.addWidget(self.camera1_combo, 0, 1)
        
        cameras_layout.addWidget(QLabel("Камера 2:"), 1, 0)
        self.camera2_combo = QComboBox()
        cameras_layout.addWidget(self.camera2_combo, 1, 1)
        
        # Параметры синхронизации
        cameras_layout.addWidget(QLabel("Порог яркости:"), 2, 0)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setMinimum(100)
        self.threshold_spin.setMaximum(255)
        self.threshold_spin.setValue(220)
        cameras_layout.addWidget(self.threshold_spin, 2, 1)
        
        cameras_layout.addWidget(QLabel("Мин. кол-во пикселей:"), 3, 0)
        self.min_pixels_spin = QSpinBox()
        self.min_pixels_spin.setMinimum(100)
        self.min_pixels_spin.setMaximum(100000)
        self.min_pixels_spin.setValue(1000)
        self.min_pixels_spin.setSingleStep(100)
        cameras_layout.addWidget(self.min_pixels_spin, 3, 1)
        
        cameras_layout.addWidget(QLabel("Макс. время ожидания (сек):"), 4, 0)
        self.max_wait_spin = QSpinBox()
        self.max_wait_spin.setMinimum(5)
        self.max_wait_spin.setMaximum(120)
        self.max_wait_spin.setValue(30)
        cameras_layout.addWidget(self.max_wait_spin, 4, 1)
        
        cameras_group.setLayout(cameras_layout)
        main_layout.addWidget(cameras_group)
        
        # Окна просмотра видео
        video_layout = QHBoxLayout()
        
        self.video_label1 = QLabel("Камера 1")
        self.video_label1.setMinimumSize(480, 360)
        self.video_label1.setAlignment(Qt.AlignCenter)
        self.video_label1.setStyleSheet("background-color: black; color: white;")
        video_layout.addWidget(self.video_label1)
        
        self.video_label2 = QLabel("Камера 2")
        self.video_label2.setMinimumSize(480, 360)
        self.video_label2.setAlignment(Qt.AlignCenter)
        self.video_label2.setStyleSheet("background-color: black; color: white;")
        video_layout.addWidget(self.video_label2)
        
        main_layout.addLayout(video_layout)
        
        # Статус
        self.status_label = QLabel("Готово к синхронизации")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Начать синхронизацию")
        self.start_button.clicked.connect(self.start_sync)
        buttons_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Остановить")
        self.stop_button.clicked.connect(self.stop_sync)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        self.save_button = QPushButton("Сохранить результаты")
        self.save_button.clicked.connect(self.save_sync)
        self.save_button.setEnabled(False)
        buttons_layout.addWidget(self.save_button)
        
        main_layout.addLayout(buttons_layout)
        
        self.setLayout(main_layout)
    
    def load_cameras(self):
        """Загружает список камер из файла cameras.txt."""
        try:
            with open("cameras.txt", "r") as f:
                camera_lines = [line.strip() for line in f if line.strip()]
            
            self.camera1_combo.clear()
            self.camera2_combo.clear()
            
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
            
            for i, (camera_url, camera_name) in enumerate(zip(cameras, camera_names)):
                self.camera1_combo.addItem(f"{camera_name}", camera_url)
                self.camera2_combo.addItem(f"{camera_name}", camera_url)
            
            # Устанавливаем разные камеры по умолчанию, если их более одной
            if len(cameras) > 1:
                self.camera2_combo.setCurrentIndex(1)
                
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Файл cameras.txt не найден. Пожалуйста, создайте файл со списком камер."
            )
    
    def update_frame(self, frame, camera_name):
        """Обновляет кадр из указанной камеры."""
        if frame is not None:
            pixmap = QPixmap.fromImage(convert_cv_qt(frame))
            if camera_name == "camera1":
                self.video_label1.setPixmap(pixmap)
            elif camera_name == "camera2":
                self.video_label2.setPixmap(pixmap)
    
    def update_status(self, message):
        """Обновляет статусную строку."""
        self.status_label.setText(message)
    
    def sync_finished(self, sync_data):
        """Обрабатывает завершение синхронизации."""
        self.sync_data = sync_data
        self.save_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def start_sync(self):
        """Запускает процесс синхронизации камер."""
        if self.camera1_combo.currentText() == self.camera2_combo.currentText():
            QMessageBox.warning(self, "Ошибка", "Выберите разные камеры для синхронизации")
            return
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        
        # Получаем параметры синхронизации
        camera1_url = self.camera1_combo.currentData()
        camera2_url = self.camera2_combo.currentData()
        flash_threshold = self.threshold_spin.value()
        min_pixels = self.min_pixels_spin.value()
        max_wait_time = self.max_wait_spin.value()
        
        # Запускаем поток синхронизации
        self.sync_thread = SyncThread(
            camera1_url, camera2_url, flash_threshold, min_pixels, max_wait_time
        )
        self.sync_thread.update_signal.connect(self.update_frame)
        self.sync_thread.status_signal.connect(self.update_status)
        self.sync_thread.finished_signal.connect(self.sync_finished)
        self.sync_thread.start()
    
    def stop_sync(self):
        """Останавливает процесс синхронизации."""
        if self.sync_thread and self.sync_thread.isRunning():
            self.sync_thread.stop()
            self.status_label.setText("Синхронизация остановлена")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def save_sync(self):
        """Сохраняет результаты синхронизации."""
        if not self.sync_data:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return
        
        try:
            # Сохранение в файл
            with open("sync_data.json", "w") as f:
                json.dump(self.sync_data, f, indent=2)
            
            # Обновление статуса синхронизации в Config
            camera1_url = self.camera1_combo.currentData()
            camera2_url = self.camera2_combo.currentData()
            self.config.update_sync_status(
                True, [camera1_url, camera2_url]
            )
            
            QMessageBox.information(
                self,
                "Успех",
                f"Данные синхронизации успешно сохранены в файл sync_data.json\n"
                f"Разница во времени: {self.sync_data['time_diff']:.6f} секунд"
            )
            
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось сохранить данные синхронизации: {str(e)}"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = SyncDialog()
    dialog.show()
    sys.exit(app.exec()) 