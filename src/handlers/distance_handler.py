from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QPixmap
from src.modules.distance_module import DistanceCalculationThread
from src.core.distance_logic import DistanceLogic
from src.utils.camera_utils import convert_cv_qt

class DistanceHandler(QObject):
    frame_signal = Signal(object, object, object)  # original_frame, processed_frame, info
    pixmap_signal = Signal(QPixmap)  # для непосредственного обновления QLabel
    log_signal = Signal(str, str, bool)  # message, color, both_logs
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.distance_thread = None
        self.cam1_frame = None
        self.cam2_frame = None
        self.last_distance_frame = None
        self.debug_counter = 0
        self.active_camera_index = 0  # По умолчанию показываем первую камеру
        
    def set_active_camera(self, index):
        """Set the active camera for display."""
        self.active_camera_index = index
        # Если есть сохраненные кадры, сразу обновляем отображение
        self.update_current_frame()
        
    def start_measurement(self, camera1_url, camera2_url, model_path):
        """Start the distance measurement process."""
        # Load calibration and sync data
        calibration_data = DistanceLogic.load_calibration_data()
        sync_data = DistanceLogic.load_sync_data()
        
        # Check camera and model selection
        if not DistanceLogic.check_camera_selection(camera1_url, camera2_url):
            return False
            
        if not DistanceLogic.check_model_selection(model_path):
            return False
            
        # Check warnings
        if not DistanceLogic.check_warnings(calibration_data, sync_data):
            return False
            
        # Stop existing thread if running
        self.stop_measurement()
        
        # Get baseline value from settings
        baseline = 10.0  # Default value
        distance_settings = self.config.get_distance_measure_settings()
        if 'baseline' in distance_settings:
            baseline = distance_settings['baseline']
        
        # Create new thread for distance measurement
        self.distance_thread = DistanceCalculationThread(
            camera1_url, camera2_url, model_path, baseline, 
            calibration_data, sync_data
        )
        
        # Connect signals
        self.distance_thread.frame_signal.connect(self.process_frames)
        self.distance_thread.error_signal.connect(self.handle_error)
        
        # Start thread
        self.distance_thread.start()
        
        # Log messages
        self.log_signal.emit(f"Измерение расстояния запущено", "black", False)
        self.log_signal.emit(f"Камера 1: {camera1_url}", "black", False)
        self.log_signal.emit(f"Камера 2: {camera2_url}", "black", False)
        self.log_signal.emit(f"Модель: {model_path}", "black", False)
        self.log_signal.emit(f"Базис: {baseline} см", "black", False)
        
        return True
        
    def stop_measurement(self):
        """Stop the distance measurement thread."""
        if self.distance_thread and self.distance_thread.isRunning():
            self.distance_thread.stop()
            self.distance_thread = None
            return True
        return False
        
    def handle_error(self, error_msg):
        """Handle errors in the distance measurement thread."""
        self.log_signal.emit(f"Ошибка: {error_msg}", "red", False)
        self.stop_measurement()
        
    def process_frames(self, original_frame, processed_frame, info):
        """Process frames from distance measurement thread."""
        if original_frame is None or processed_frame is None:
            return  # Skip processing if one of the frames is missing
            
        # Save frames for camera switching
        self.cam1_frame = original_frame.copy()
        self.cam2_frame = processed_frame.copy()
        
        # Отправляем сигнал с кадрами для обработки в Widget
        self.frame_signal.emit(original_frame, processed_frame, info)
        
        # Обновляем текущий отображаемый кадр в зависимости от выбранной камеры
        self.update_current_frame()
        
        self.debug_counter += 1
        
        # Process detection info every 30 frames
        if detections := info.get('detections', {}) and self.debug_counter % 30 == 0:
            self.log_signal.emit("Распознанные объекты:", "black", False)
            
            # Add only the first 5 objects to save space in the log
            for i, (obj_id, obj_data) in enumerate(list(detections.items())[:5]):
                cls_name = obj_data['class']
                distance = obj_data['distance']
                confidence = obj_data['confidence']
                
                # Determine color based on distance
                color = "green"
                if distance < 2:  # less than 2 meters
                    color = "red"
                elif distance < 5:  # 2-5 meters
                    color = "orange"
                
                # Add message with formatting
                self.log_signal.emit(
                    f'<b>{cls_name}</b> (ID: {obj_id.split("_")[1]}): '
                    f'{distance:.2f} м (увер.: {confidence:.2f})',
                    color, False
                )
        
    def get_frame(self, camera_index):
        """Get frame for the specified camera index."""
        if camera_index == 0 and self.cam1_frame is not None:
            return self.cam1_frame
        elif camera_index == 1 and self.cam2_frame is not None:
            return self.cam2_frame
        return None
        
    def refresh_stream(self):
        """Force refresh of the camera stream."""
        return self.cam1_frame is not None and self.cam2_frame is not None 

    def update_current_frame(self):
        """Update the current frame based on the active camera index."""
        frame = self.get_frame(self.active_camera_index)
        if frame is not None:
            # Конвертируем кадр для отображения в Qt формат
            qt_img = convert_cv_qt(frame)
            
            # Создаем QPixmap из QImage
            pixmap = QPixmap.fromImage(qt_img)
            
            # Отправляем сигнал для непосредственного обновления интерфейса
            self.pixmap_signal.emit(pixmap)
            return True
        return False 