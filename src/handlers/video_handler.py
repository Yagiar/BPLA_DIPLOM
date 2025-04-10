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
        self.model_path = self.config.get_last_model()

    def select_camera(self, camera_url):
        """Select and start the video stream for the chosen camera."""
        self.selected_camera_url = camera_url
        self.stop_video_stream()

        # Check camera availability
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            cap.release()
            return False
        cap.release()

        # Get the latest model path from config
        self.model_path = self.config.get_last_model()

        # Start new video thread
        self.thread = VideoThread(
            camera_url,
            conf=self.config.get_detection_settings()['confidence_threshold'],
            iou=self.config.get_detection_settings()['iou_threshold'],
            device='cpu',
            half=False,
            fps=30
        )
        # Connect the signal with a conversion slot
        self.thread.change_pixmap_signal.connect(self.update_video_frame)
        self.thread.detection_signal.connect(self.log_detection)

        # Load model if selected
        if self.model_path:
            self.thread.set_model(self.model_path)

        self.thread.start()
        self.config.set_last_camera(camera_url)
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