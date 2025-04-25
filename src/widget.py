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
import supervision as sv

# Импорты из utils
from src.utils.camera_utils import convert_cv_qt
from src.utils.camera_loader import CameraLoader

# Импорты из ui
from src.ui.settings_dialog import SettingsDialog
from src.ui.ui_components import UIComponentsFactory
from src.ui.app_styles import AppStyles

# Импорты из core
from src.core.config import Config
from src.core.distance_logic import DistanceLogic

# Импорты из modules
from src.modules.calibration_module import CalibrationDialog
from src.modules.sync_module import SyncDialog
from src.modules.distance_module import DistanceCalculationThread

# Импорты из handlers
from src.handlers.video_handler import VideoHandler
from src.handlers.log_manager import LogManager
from src.handlers.distance_handler import DistanceHandler


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Мониторинг БПЛА")
        self.setMinimumSize(1000, 700)

        # Initialize configuration
        self.config = Config()
        
        # Initialize UI components
        self.init_ui()
        
        # Initialize handlers
        self.init_handlers()
        
        # Load cameras
        self.load_cameras()
        
        # Check calibration and sync status
        self.update_calibration_sync_status()

    def init_ui(self):
        """Initialize the UI components."""
        # Set application style
        self.setStyleSheet(AppStyles.get_main_stylesheet())
        
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Add title
        self.title_label = UIComponentsFactory.create_title_label("Система мониторинга БПЛА")
        main_layout.addWidget(self.title_label)
        
        # Create mode selector
        self.mode_group, self.mode_detection, self.mode_distance = UIComponentsFactory.create_mode_selector(
            self.change_mode
        )
        main_layout.addWidget(self.mode_group)
        
        # Create stacked widget for different modes
        self.stacked_widget = QStackedWidget()
        
        # Page 1: Detection mode
        self.detection_widget = QWidget()
        self.video_label = UIComponentsFactory.create_video_label()
        self.log_text_edit = UIComponentsFactory.create_log_text_edit()
        
        # Create camera buttons container
        self.cameras_widget = QWidget()
        self.cameras_container = QVBoxLayout(self.cameras_widget)
        self.cameras_container.setSpacing(5)
        self.cameras_container.setContentsMargins(5, 5, 5, 5)
        self.cameras_container.addStretch()
        
        # Create detection layout
        detection_layout = UIComponentsFactory.create_detection_layout(
            self.video_label, 
            self.log_text_edit,
            self.cameras_widget,
            self.select_model,
            self.show_settings
        )
        self.detection_widget.setLayout(detection_layout)
        
        # Page 2: Distance measurement mode
        self.distance_widget = QWidget()
        self.distance_video_label = UIComponentsFactory.create_video_label("Выберите две камеры для измерения расстояния")
        self.distance_log_text_edit = UIComponentsFactory.create_log_text_edit()
        
        # Create distance layout
        distance_layout, self.distance_widgets = UIComponentsFactory.create_distance_layout(
            self.distance_video_label,
            self.distance_log_text_edit,
            self.select_model,
            self.show_settings,
            self.start_distance_measurement,
            self.stop_distance_measurement,
            self.open_calibration_dialog,
            self.open_sync_dialog
        )
        self.distance_widget.setLayout(distance_layout)
        
        # Store references to important widgets
        self.cam1_combo = self.distance_widgets['cam1_combo']
        self.cam2_combo = self.distance_widgets['cam2_combo']
        self.active_cam_combo = self.distance_widgets['active_cam_combo']
        self.start_distance_button = self.distance_widgets['start_distance_button']
        self.stop_distance_button = self.distance_widgets['stop_distance_button']
        self.calibration_status_label = self.distance_widgets['calibration_status_label']
        self.sync_status_label = self.distance_widgets['sync_status_label']
        
        # Connect active camera combo
        self.active_cam_combo.currentIndexChanged.connect(self.on_camera_switch)
        
        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.detection_widget)
        self.stacked_widget.addWidget(self.distance_widget)
        
        main_layout.addWidget(self.stacked_widget)
        
        self.setLayout(main_layout)

    def init_handlers(self):
        """Initialize the handlers."""
        # Mode and model
        self.mode = "detection"  # default mode: detection, distance
        self.model_path = self.config.get_model_path()
        
        # Initialize video handler
        self.video_handler = VideoHandler(self.config)
        self.video_handler.change_pixmap_signal.connect(self.update_video_frame)
        # Connect detection signal to log message
        self.video_handler.log_detection = lambda msg, color: self.log_message(msg, color)
        
        # Initialize log manager
        self.log_manager = LogManager(self.log_text_edit, self.distance_log_text_edit)
        
        # Initialize distance handler
        self.distance_handler = DistanceHandler(self.config)
        self.distance_handler.frame_signal.connect(self.update_distance_frame)
        # Подключаем также pixmap_signal для непосредственного обновления QLabel
        self.distance_handler.pixmap_signal.connect(self.update_video_frame)
        self.distance_handler.log_signal.connect(self.log_message)

    def update_calibration_sync_status(self):
        """Update the calibration and synchronization status labels."""
        # Update calibration status
        calibration_status = self.config.get_calibration_status()
        if calibration_status.get('calibrated', False):
            self.calibration_status_label.setText("Калибровка: ✅")
        else:
            self.calibration_status_label.setText("Калибровка: ❌")
            
        # Update sync status
        sync_status = self.config.get_sync_status()
        if sync_status.get('synced', False):
            self.sync_status_label.setText("Синхронизация: ✅")
        else:
            self.sync_status_label.setText("Синхронизация: ❌")

    def change_mode(self, mode):
        """Change the application mode."""
        if mode == self.mode:
            return
            
        # Stop current video streams
        self.video_handler.stop_video_stream()
        self.distance_handler.stop_measurement()
            
        self.mode = mode
        
        if mode == "detection":
            self.stacked_widget.setCurrentIndex(0)
            self.log_message("Режим распознавания и трекинга активирован")
        elif mode == "distance":
            self.stacked_widget.setCurrentIndex(1)
            self.log_message("Режим измерения расстояния активирован", both_logs=True)
            
            # Check calibration and synchronization
            calibration_status = self.config.get_calibration_status()
            sync_status = self.config.get_sync_status()
            
            if not calibration_status.get('calibrated', False):
                self.log_message("⚠️ Камеры не откалиброваны! Рекомендуется выполнить калибровку.", "orange", both_logs=True)
                
            if not sync_status.get('synced', False):
                self.log_message("⚠️ Камеры не синхронизированы! Рекомендуется выполнить синхронизацию.", "orange", both_logs=True)

    def open_calibration_dialog(self):
        """Open the camera calibration dialog."""
        dialog = CalibrationDialog(self)
        if dialog.exec():
            # Update status after calibration
            self.update_calibration_sync_status()
            self.log_message("Калибровка камер завершена", "green")

    def open_sync_dialog(self):
        """Open the camera synchronization dialog."""
        dialog = SyncDialog(self)
        if dialog.exec():
            # Update status after synchronization
            self.update_calibration_sync_status()
            self.log_message("Синхронизация камер завершена", "green")

    def start_distance_measurement(self):
        """Start the distance measurement process."""
        # Get camera URLs from combo boxes
        camera1_url = self.cam1_combo.currentData()
        camera2_url = self.cam2_combo.currentData()
        
        # Always get the latest model path from config
        self.model_path = self.config.get_model_path()
        
        if self.distance_handler.start_measurement(camera1_url, camera2_url, self.model_path):
            # Update UI
            self.start_distance_button.setEnabled(False)
            self.stop_distance_button.setEnabled(True)

    def stop_distance_measurement(self):
        """Stop the distance measurement process."""
        self.distance_handler.stop_measurement()
        
        # Clear image
        self.distance_video_label.setText("Выберите две камеры для измерения расстояния")
        self.distance_video_label.setPixmap(QPixmap())
        
        self.log_message("Измерение расстояния остановлено")
        self.start_distance_button.setEnabled(True)
        self.stop_distance_button.setEnabled(False)

    def show_settings(self):
        """Open the settings dialog."""
        dialog = SettingsDialog(self, self.model_path)
        
        # Load current settings from model section instead of detection
        model_settings = self.config.get_model_settings()
        settings = {
            'conf': model_settings['conf'],
            'iou': model_settings['iou'],
            'device': model_settings['device'],
            'half': model_settings['half'],
            'fps': 30  # Default value for FPS
        }
        dialog.set_settings(settings)
        
        if dialog.exec():
            # Get and save new settings
            new_settings = dialog.get_settings()
            
            # Update model settings instead of detection settings
            self.config.set_model_settings(
                conf=new_settings.get('conf'),
                iou=new_settings.get('iou'),
                device=new_settings.get('device'),
                half=new_settings.get('half')
            )
            
            # If video stream is running, apply new settings
            if self.video_handler.thread and self.video_handler.thread.isRunning():
                self.video_handler.thread.update_settings(new_settings)
                self.log_message("Настройки успешно обновлены", "green", both_logs=True)
            else:
                self.log_message("Настройки сохранены и будут применены при запуске видеопотока", "blue", both_logs=True)

    def log_message(self, message, color="black", both_logs=False):
        """Log a message to the appropriate log panel."""
        self.log_manager.log_message(message, color, both_logs)

    def load_cameras(self):
        """Load the camera list from cameras.txt and create buttons for them."""
        cameras, camera_names = CameraLoader.load_from_file("cameras.txt")
        
        if cameras is None:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Файл cameras.txt не найден. Пожалуйста, создайте файл со списком камер."
            )
            sys.exit(1)
            
        if not cameras:
            QMessageBox.warning(self, "Ошибка", "Файл cameras.txt пуст.")
            return
        
        # Clear existing camera lists
        # Remove all widgets from the container
        while self.cameras_container.count():
            item = self.cameras_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        # Populate camera combo boxes
        CameraLoader.populate_comboboxes(cameras, camera_names, self.cam1_combo, self.cam2_combo)
        
        # Add camera buttons to the container
        self.log_message("Камеры успешно загружены.", "black")
        
        for i, (camera_url, camera_name) in enumerate(zip(cameras, camera_names)):
            # Create camera button with icon and name from file
            btn = UIComponentsFactory.create_camera_button(camera_name, camera_url, self.select_camera, i)
            self.cameras_container.addWidget(btn)
                
        # Add stretch at the end of camera list
        self.cameras_container.addStretch()
            
        # Set different default cameras for distance mode
        if len(cameras) > 1:
            self.cam2_combo.setCurrentIndex(1)
            
        # Get last distance settings from configuration
        if hasattr(self.config, 'get_distance_measure_settings'):
            distance_settings = self.config.get_distance_measure_settings()
            if 'cameras' in distance_settings and len(distance_settings['cameras']) >= 2:
                # Set saved cameras
                cam1 = distance_settings['cameras'][0]
                cam2 = distance_settings['cameras'][1]
                
                # Find saved cameras by URL in combo box data
                cam1_index = CameraLoader.find_camera_index_by_url(self.cam1_combo, cam1)
                if cam1_index >= 0:
                    self.cam1_combo.setCurrentIndex(cam1_index)
                    
                cam2_index = CameraLoader.find_camera_index_by_url(self.cam2_combo, cam2)
                if cam2_index >= 0:
                    self.cam2_combo.setCurrentIndex(cam2_index)

    @Slot()
    def select_model(self):
        """Select a YOLO model (.pt file)."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите модель YOLO",
            "",
            "YOLO Model (*.pt)"
        )
        if file_name:
            self.model_path = file_name
            
            # Сохраним путь к модели в конфигурацию
            self.config.set_model_path(file_name)
            
            self.log_message(f"Выбрана модель: {file_name}", "blue", both_logs=True)
            
            # If video stream is running, update the model
            if self.video_handler.thread and self.video_handler.thread.isRunning():
                if self.video_handler.thread.set_model(self.model_path):
                    self.log_message("Модель успешно загружена в режиме распознавания", "green", both_logs=True)
                else:
                    self.log_message("Ошибка при загрузке модели в режиме распознавания", "red", both_logs=True)
            
            # Also update distance handler model if it's running
            if self.distance_handler.distance_thread and self.distance_handler.distance_thread.isRunning():
                # Need to restart the distance measurement with the new model
                self.log_message("Перезапуск измерения расстояния с новой моделью...", "blue", False)
                
                # Get current camera URLs
                camera1_url = self.cam1_combo.currentData()
                camera2_url = self.cam2_combo.currentData()
                
                # Stop and restart with new model
                self.distance_handler.stop_measurement()
                if self.distance_handler.start_measurement(camera1_url, camera2_url, self.model_path):
                    self.log_message("Модель успешно загружена в режиме измерения расстояния", "green", False)
                else:
                    self.log_message("Ошибка при загрузке модели в режиме измерения расстояния", "red", False)

    @Slot()
    def select_camera(self, camera_url):
        """Select and start the video stream for the chosen camera."""
        self.log_message(f"Выбрана камера: {camera_url}", "black")
        if not self.video_handler.select_camera(camera_url):
            self.log_message(f"Камера {camera_url} недоступна.", "red")

    @Slot(QPixmap)
    def update_video_frame(self, pixmap):
        """Update the video frame in the UI."""
        target_label = self.video_label if self.mode == "detection" else self.distance_video_label
        scaled_pixmap = pixmap.scaled(
            target_label.width(), 
            target_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        target_label.setPixmap(scaled_pixmap)
        target_label.setAlignment(Qt.AlignCenter)
        
    @Slot(object, object, object)
    def update_distance_frame(self, original_frame, processed_frame, info):
        """Update the frame and distance information in the UI."""
        # Determine which camera to display
        display_index = self.active_cam_combo.currentIndex()
        
        # Select frame based on active camera index
        # Index 0 - Camera 1 (original_frame), Index 1 - Camera 2 (processed_frame)
        frame_to_display = original_frame if display_index == 0 else processed_frame
        
        # Convert frame for display in Qt format
        qt_img = convert_cv_qt(frame_to_display)
        
        # Create QPixmap from QImage
        pixmap = QPixmap.fromImage(qt_img)
        
        # Scale image while preserving aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.distance_video_label.width(), 
            self.distance_video_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # Display on label
        self.distance_video_label.setPixmap(scaled_pixmap)
        self.distance_video_label.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        """Handle window resize event."""
        super().resizeEvent(event)
        
        # Update video display proportionally if there's a frame
        if self.mode == "detection" and self.video_handler.thread and self.video_handler.thread.isRunning():
            if hasattr(self.video_handler.thread, 'last_frame') and self.video_handler.thread.last_frame is not None:
                self.update_video_frame(QPixmap.fromImage(convert_cv_qt(self.video_handler.thread.last_frame)))
        # In distance mode, update from the appropriate camera frame if available
        elif self.mode == "distance" and self.distance_handler.distance_thread and self.distance_handler.distance_thread.isRunning():
            frame = self.distance_handler.get_frame(self.active_cam_combo.currentIndex())
            if frame is not None:
                qt_img = convert_cv_qt(frame)
                pixmap = QPixmap.fromImage(qt_img)
                self.update_video_frame(pixmap)

    def closeEvent(self, event):
        """Handle window close event."""
        # Stop main recognition thread
        self.video_handler.stop_video_stream()
        
        # Stop distance measurement thread if running
        self.distance_handler.stop_measurement()
        
        event.accept()

    def on_camera_switch(self, index):
        """Handle camera switch in dropdown list."""
        self.log_message(f"Переключение на камеру: {index + 1}", "black", False)
        
        # Обновляем индекс активной камеры в distance_handler
        self.distance_handler.set_active_camera(index)
        
        # If we have saved frames, switch display immediately
        frame = self.distance_handler.get_frame(index)
        if frame is not None:
            self.log_message(f"Отображение переключено на камеру {index + 1}", "green", False)
    
    def refresh_video_stream(self):
        """Force refresh of the video stream."""
        if self.distance_handler.refresh_stream():
            current_camera_index = self.active_cam_combo.currentIndex()
            self.log_message(f"Принудительное обновление потока для камеры {current_camera_index + 1}", "blue", False)
            
            frame = self.distance_handler.get_frame(current_camera_index)
            if frame is not None:
                qt_img = convert_cv_qt(frame)
                pixmap = QPixmap.fromImage(qt_img)
                self.update_video_frame(pixmap)
            else:
                self.log_message("Сохраненные кадры не найдены, перезапуск потока...", "orange", False)
                self.stop_distance_measurement()
                self.start_distance_measurement()

# Application entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
