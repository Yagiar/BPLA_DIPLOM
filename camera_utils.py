import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from ultralytics import YOLO
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack


def convert_cv_qt(cv_img):
    """Конвертирует изображение OpenCV в QImage."""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)


class VideoThread(QThread):
    """Поток для захвата видеопотока."""
    change_pixmap_signal = Signal(np.ndarray)
    detection_signal = Signal(str, str)

    def __init__(self, camera_url, conf=0.25, iou=0.45, device='cpu', half=False, fps=30):
        super().__init__()
        self.camera_url = camera_url
        self.running = True
        self.model = None
        self.tracker = None
        
        # Настройки модели и трекера
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half
        self.fps = fps

    def set_model(self, model_path):
        """Загружает модель YOLO."""
        try:
            self.model = YOLO(model_path)
            self.tracker = ByteTrack()
            return True
        except Exception as e:
            self.detection_signal.emit(f"Ошибка загрузки модели: {e}", "red")
            return False

    def update_settings(self, settings):
        """Обновляет настройки модели и трекера."""
        if 'conf' in settings:
            self.conf = settings['conf']
        if 'iou' in settings:
            self.iou = settings['iou']
        if 'device' in settings:
            self.device = settings['device']
        if 'half' in settings:
            self.half = settings['half']
        if 'fps' in settings:
            self.fps = settings['fps']

    def run(self):
        """Запускает обработку видеопотока."""
        cap = cv2.VideoCapture(self.camera_url)
        frame_delay = 1.0 / self.fps

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.detection_signal.emit(f"Ошибка чтения кадра с камеры {self.camera_url}", "red")
                break

            try:
                if self.model:
                    # Получаем результаты детекции
                    results = self.model.predict(
                        frame,
                        conf=self.conf,
                        iou=self.iou,
                        device=self.device,
                        half=self.half
                    )[0]

                    # Конвертируем результаты в формат для трекера
                    detections = sv.Detections.from_ultralytics(results)

                    if len(detections) > 0:
                        # Применяем трекер
                        tracked_detections = self.tracker.update_with_detections(detections)

                        # Отрисовываем боксы и ID
                        box_annotator = sv.BoxAnnotator()
                        frame = box_annotator.annotate(
                            scene=frame,
                            detections=tracked_detections
                        )

                        # Отправляем сообщение о детекции
                        self.detection_signal.emit(
                            f"Обнаружено объектов: {len(tracked_detections)}",
                            "blue"
                        )

            except Exception as e:
                self.detection_signal.emit(f"Ошибка обработки кадра: {e}", "red")

            # Отправляем кадр для отображения
            self.change_pixmap_signal.emit(frame)
            
            # Задержка для поддержания заданного FPS
            cv2.waitKey(int(frame_delay * 1000))

        cap.release()

    def stop(self):
        """Останавливает поток."""
        self.running = False
        self.wait() 