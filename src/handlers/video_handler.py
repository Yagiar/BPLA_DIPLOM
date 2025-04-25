import cv2
from PySide6.QtCore import Signal, QObject
from PySide6.QtGui import QPixmap
from src.utils.camera_utils import convert_cv_qt, VideoThread

class VideoHandler(QObject):
    change_pixmap_signal = Signal(QPixmap)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.thread = None
        self.selected_camera_url = None
        self.model_path = self.config.get_model_path()

    def select_camera(self, camera_url):
        """Выбирает камеру и запускает видеопоток."""
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            
        # Загружаем настройки из секции model вместо detection
        model_settings = self.config.get_model_settings()
        
        # Создаем новый поток для обработки видео
        self.thread = VideoThread(
            camera_url,
            conf=model_settings['conf'],
            iou=model_settings['iou'],
            device=model_settings['device'],
            half=model_settings['half'],
            fps=30
        )
        
        # Подключаем сигналы
        self.thread.change_pixmap_signal.connect(self.update_video_frame)
        self.thread.detection_signal.connect(self.log_detection)
        
        # Загружаем модель, если путь задан
        model_path = self.config.get_model_path()
        if model_path:
            if not self.thread.set_model(model_path):
                self.log_detection(f"Ошибка загрузки модели из {model_path}", "red")
                return False
        
        # Запускаем поток
        self.thread.start()
        self.log_detection(f"Выбрана камера: {camera_url}", "blue")
        return True

    def stop_video_stream(self):
        """Stop the current video stream if running."""
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread = None

    def update_video_frame(self, frame):
        """Update the video frame in the UI."""
        if frame is not None:
            qt_img = convert_cv_qt(frame)
            pixmap = QPixmap.fromImage(qt_img)
            self.change_pixmap_signal.emit(pixmap)
    
    def log_detection(self, message, color):
        """Pass through detection messages to be logged."""
        # This will be connected to the logging system from the widget class
        pass 