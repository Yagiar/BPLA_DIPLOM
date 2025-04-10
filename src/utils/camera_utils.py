import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from ultralytics import YOLO
import supervision as sv


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
        
        # Настройки модели и трекера
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half
        self.fps = fps
        
        # Аннотаторы
        self.model = None
        self.tracker = None
        self.box_annotator = None
        self.label_annotator = None
        self.trace_annotator = None

    def set_model(self, model_path):
        """Загружает модель YOLO и инициализирует аннотаторы."""
        try:
            self.model = YOLO(model_path)
            self.tracker = sv.ByteTrack()
            self.box_annotator = sv.BoundingBoxAnnotator()
            self.label_annotator = sv.LabelAnnotator()
            self.trace_annotator = sv.TraceAnnotator()
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
                if self.model and self.tracker:
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
                    
                    # Обновляем трекер
                    detections = self.tracker.update_with_detections(detections)

                    # Создаем подписи с ID и названием класса
                    labels = [
                        f"#{tracker_id} {results.names[class_id]}"
                        for class_id, tracker_id 
                        in zip(detections.class_id, detections.tracker_id)
                    ]

                    # Аннотируем кадр
                    annotated_frame = self.box_annotator.annotate(
                        frame.copy(), 
                        detections=detections
                    )
                    annotated_frame = self.label_annotator.annotate(
                        annotated_frame, 
                        detections=detections, 
                        labels=labels
                    )
                    annotated_frame = self.trace_annotator.annotate(
                        annotated_frame, 
                        detections=detections
                    )

                    # Отправляем сообщения о каждом обнаруженном объекте
                    for idx in range(len(detections)):
                        class_id = int(detections.class_id[idx])
                        tracker_id = detections.tracker_id[idx]
                        class_name = results.names[class_id]
                        
                        # Отправляем сообщение о каждом обнаруженном объекте
                        self.detection_signal.emit(
                            f"Обнаружен {class_name} (ID: {tracker_id})",
                            "blue"
                        )

                    # Общее сообщение о количестве объектов
                    self.detection_signal.emit(
                        f"Обнаружено объектов: {len(detections)}",
                        "blue"
                    )

                    # Отправляем кадр для отображения
                    self.change_pixmap_signal.emit(annotated_frame)

            except Exception as e:
                self.detection_signal.emit(f"Ошибка обработки кадра: {e}", "red")
                import traceback
                traceback.print_exc()
            
            # Задержка для поддержания заданного FPS
            cv2.waitKey(int(frame_delay * 1000))

        cap.release()

    def stop(self):
        """Останавливает поток."""
        self.running = False
        self.wait() 