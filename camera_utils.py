import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from ultralytics import YOLO
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack


def convert_cv_qt(cv_img):
    """Конвертирует изображение OpenCV (BGR) в QPixmap для отображения в QLabel."""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return qt_img


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
            self.tracker = ByteTrack(frame_rate=30)
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
                    if self.tracker is not None:
                        try:
                            # Получаем результаты детекции YOLO
                            results = self.model(frame)[0]
                            
                            # Преобразуем результаты в объект Detections из supervision
                            detections = sv.Detections.from_ultralytics(results)
                            
                            # Обновляем трекер, используя метод update_with_detections
                            tracks = self.tracker.update_with_detections(detections)
                            
                            # Отрисовываем результаты трекинга
                            for idx in range(len(tracks.xyxy)):
                                x1, y1, x2, y2 = map(int, tracks.xyxy[idx])
                                tracker_id = tracks.tracker_id[idx]
                                class_id = int(tracks.class_id[idx])
                                class_name = results.names[class_id]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"{class_name} ID:{tracker_id}"
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                self.detection_signal.emit(f"Обнаружен объект: {class_name} (ID: {tracker_id})")
                        except Exception as e:
                            self.detection_signal.emit(f"Ошибка при обработке кадра: {str(e)}")
                    else:
                        self.detection_signal.emit("Трекер не инициализирован, пропускаю обработку кадра")
                self.change_pixmap_signal.emit(frame)
            else:
                self.change_pixmap_signal.emit(None)
                break
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait() 