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
from bytetrack import BYTETracker


def convert_cv_qt(cv_img):
    """Конвертирует изображение OpenCV (BGR) в QPixmap для отображения в QLabel."""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_img)


class VideoThread(QThread):
    """Поток для захвата видеопотока."""
    change_pixmap_signal = Signal(object)
    detection_signal = Signal(str)

    def __init__(self, camera_url):
        super().__init__()
        self.camera_url = camera_url
        self._run_flag = True
        self.model = None
        self.tracker = None

    def set_model(self, model_path):
        """Установка модели YOLO и инициализация трекера."""
        try:
            self.model = YOLO(model_path)
            self.tracker = BYTETracker(
                track_thresh=0.25,
                track_buffer=30,
                match_thresh=0.8,
                frame_rate=30
            )
            return True
        except Exception as e:
            self.detection_signal.emit(f"Ошибка загрузки модели: {str(e)}")
            return False

    def run(self):
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            self.change_pixmap_signal.emit(None)
            return

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                if self.model is not None:
                    try:
                        # Получаем результаты детекции YOLO
                        results = self.model(frame)[0]
                        
                        # Подготавливаем данные для трекера
                        detections = []
                        for box, conf, cls in zip(results.boxes.xyxy.cpu().numpy(), 
                                                results.boxes.conf.cpu().numpy(),
                                                results.boxes.cls.cpu().numpy()):
                            x1, y1, x2, y2 = box
                            detections.append([x1, y1, x2, y2, conf, cls])
                        
                        if len(detections) > 0:
                            detections = np.array(detections)
                            # Обновляем трекер
                            tracks = self.tracker.update(detections, frame)
                            
                            # Отрисовываем результаты трекинга
                            for track in tracks:
                                track_id = int(track[4])
                                class_id = int(track[5])
                                x1, y1, x2, y2 = map(int, track[:4])
                                
                                # Получаем имя класса из модели
                                class_name = results.names[class_id]
                                
                                # Рисуем бокс и метку
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"{class_name} ID:{track_id}"
                                cv2.putText(frame, label, (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                
                                # Отправляем информацию в лог
                                self.detection_signal.emit(
                                    f"Обнаружен объект: {class_name} (ID: {track_id})"
                                )
                    
                    except Exception as e:
                        self.detection_signal.emit(f"Ошибка при обработке кадра: {str(e)}")
                
                self.change_pixmap_signal.emit(frame)
            else:
                self.change_pixmap_signal.emit(None)
                break
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Мониторинг БПЛА")

        # Флаг успешного подключения камеры
        self.connected = False

        # Кнопка выбора модели
        self.model_button = QPushButton("Выбор модели YOLO")
        self.model_button.clicked.connect(self.select_model)

        # Контейнер для списка камер
        self.cameras_container = QVBoxLayout()

        # Создадим виджет-обёртку для кнопок камер
        self.cameras_widget = QWidget()
        self.cameras_widget.setLayout(self.cameras_container)

        # Окно с видео
        self.video_label = QLabel("Окно с видео")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")

        # Панель логов
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        # Компоновка интерфейса
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.model_button)
        control_layout.addWidget(self.cameras_widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.log_text_edit)

        layout = QHBoxLayout()
        layout.addLayout(main_layout)
        layout.addLayout(control_layout)

        self.setLayout(layout)

        # Переменные для работы с видеопотоком
        self.thread = None
        self.selected_camera_url = None
        self.model_path = None

        # Загрузка камер из файла
        self.load_cameras()

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
        
        # Запуск нового потока
        self.thread = VideoThread(camera_url)
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
        if frame is None:
            self.log_message(f"Поток с камеры {self.selected_camera_url} упал.", "red")
            return
        
        if not self.connected:
            self.log_message(f"Подключение к камере {self.selected_camera_url} успешно.", "green")
            self.connected = True
            
        try:
            qt_pixmap = convert_cv_qt(frame)
            self.video_label.setPixmap(qt_pixmap)
        except Exception as e:
            self.log_message(f"Ошибка отображения кадра: {e}", "red")

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
