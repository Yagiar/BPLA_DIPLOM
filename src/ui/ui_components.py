from PySide6.QtWidgets import (
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QGroupBox, QRadioButton, QButtonGroup,
    QComboBox, QScrollArea, QSizePolicy, QWidget, QLineEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QIntValidator

class UIComponentsFactory:
    """Factory class for creating and managing UI components."""
    
    @staticmethod
    def create_title_label(text):
        """Create a title label with the specified text."""
        title_label = QLabel(text)
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        return title_label
    
    @staticmethod
    def create_video_label(text="Ожидание видеопотока..."):
        """Create a video label with the specified text."""
        video_label = QLabel(text)
        video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_label.setMinimumSize(320, 240)
        video_label.setObjectName("video_label")
        video_label.setAlignment(Qt.AlignCenter)
        video_label.setScaledContents(False)
        return video_label
    
    @staticmethod
    def create_log_text_edit():
        """Create a log text edit widget."""
        log_text_edit = QTextEdit()
        log_text_edit.setReadOnly(True)
        log_text_edit.setMinimumHeight(100)
        log_text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        return log_text_edit
    
    @staticmethod
    def create_detection_layout(video_label, log_text_edit, camera_buttons_container, on_select_model, on_show_settings):
        """Create the detection mode layout."""
        detection_layout = QHBoxLayout()
        
        # Video panel
        video_layout = QVBoxLayout()
        video_layout.addWidget(video_label, 7)
        video_layout.addWidget(log_text_edit, 3)
        
        # Control panel
        control_layout = QVBoxLayout()
        
        # Buttons group
        buttons_group = QGroupBox("Управление")
        buttons_layout = QVBoxLayout()
        
        # First row of buttons
        main_buttons_layout = QHBoxLayout()
        
        model_button = QPushButton("📁 Выбор модели YOLO")
        model_button.clicked.connect(on_select_model)
        model_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        settings_button = QPushButton("⚙️ Настройки")
        settings_button.clicked.connect(on_show_settings)
        settings_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        main_buttons_layout.addWidget(model_button)
        main_buttons_layout.addWidget(settings_button)
        
        buttons_layout.addLayout(main_buttons_layout)
        buttons_group.setLayout(buttons_layout)
        control_layout.addWidget(buttons_group)
        
        # Cameras group
        cameras_group = QGroupBox("Доступные источники видео")
        cameras_layout = QVBoxLayout()
        
        cameras_scroll = QScrollArea()
        cameras_scroll.setWidgetResizable(True)
        cameras_scroll.setMinimumHeight(150)
        cameras_scroll.setWidget(camera_buttons_container)
        
        cameras_layout.addWidget(cameras_scroll)
        cameras_group.setLayout(cameras_layout)
        control_layout.addWidget(cameras_group)
        control_layout.addStretch()
        
        detection_layout.addLayout(video_layout, 7)
        detection_layout.addLayout(control_layout, 3)
        
        return detection_layout
    
    @staticmethod
    def create_distance_layout(video_label, log_text_edit, on_select_model, on_show_settings, 
                               on_start_distance, on_stop_distance, on_calibration, on_sync):
        """Create the distance measurement mode layout."""
        distance_layout = QHBoxLayout()
        
        # Video panel
        distance_video_layout = QVBoxLayout()
        distance_video_layout.addWidget(video_label, 7)
        distance_video_layout.addWidget(log_text_edit, 3)
        
        # Control panel
        distance_control_layout = QVBoxLayout()
        
        # Camera selection group
        camera_selection_group = QGroupBox("Выбор камер для измерения")
        camera_selection_layout = QVBoxLayout()
        
        # Camera 1 selection
        cam1_layout = QHBoxLayout()
        cam1_layout.addWidget(QLabel("Камера 1:"))
        cam1_combo = QComboBox()
        cam1_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        cam1_layout.addWidget(cam1_combo)
        camera_selection_layout.addLayout(cam1_layout)
        
        # Camera 2 selection
        cam2_layout = QHBoxLayout()
        cam2_layout.addWidget(QLabel("Камера 2:"))
        cam2_combo = QComboBox()
        cam2_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        cam2_layout.addWidget(cam2_combo)
        camera_selection_layout.addLayout(cam2_layout)
        
        # Active camera selection
        active_cam_layout = QHBoxLayout()
        active_cam_layout.addWidget(QLabel("Показывать:"))
        active_cam_combo = QComboBox()
        active_cam_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        active_cam_combo.addItem("Камера 1")
        active_cam_combo.addItem("Камера 2")
        active_cam_layout.addWidget(active_cam_combo)
        camera_selection_layout.addLayout(active_cam_layout)
        
        # Baseline adjustment input
        adjustment_layout = QHBoxLayout()
        adjustment_layout.addWidget(QLabel("Юстировка (px):"))
        adjustment_text = QLineEdit()
        adjustment_text.setPlaceholderText("Введите значение в пикселях")
        adjustment_text.setValidator(QIntValidator(1, 1000))
        adjustment_text.setText("1")
        adjustment_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        adjustment_layout.addWidget(adjustment_text)
        camera_selection_layout.addLayout(adjustment_layout)
        # Альтернативный вариант с сохранением стиля приложения
        adjustment_text.setStyleSheet("background-color: #e8f4f8; color: #333333; border: 1px solid #4caf50; border-radius: 4px; padding: 4px;")
        
        # Measurement buttons
        measurement_buttons_layout = QHBoxLayout()
        
        start_distance_button = QPushButton("▶️ Запустить измерение")
        start_distance_button.clicked.connect(on_start_distance)
        start_distance_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        stop_distance_button = QPushButton("⏹️ Остановить")
        stop_distance_button.clicked.connect(on_stop_distance)
        stop_distance_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        stop_distance_button.setEnabled(False)
        
        measurement_buttons_layout.addWidget(start_distance_button)
        measurement_buttons_layout.addWidget(stop_distance_button)
        
        camera_selection_layout.addLayout(measurement_buttons_layout)
        camera_selection_group.setLayout(camera_selection_layout)
        distance_control_layout.addWidget(camera_selection_group)
        
        # Status group
        status_group = QGroupBox("Статус")
        status_layout = QVBoxLayout()
        
        calibration_status_label = QLabel("Калибровка: ❌")
        sync_status_label = QLabel("Синхронизация: ❌")
        
        status_layout.addWidget(calibration_status_label)
        status_layout.addWidget(sync_status_label)
        status_group.setLayout(status_layout)
        
        distance_control_layout.addWidget(status_group)
        
        # Tools group
        tools_group = QGroupBox("Инструменты")
        tools_layout = QVBoxLayout()
        
        calibration_button = QPushButton("🔍 Калибровка камер")
        calibration_button.clicked.connect(on_calibration)
        calibration_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        sync_button = QPushButton("⏱️ Синхронизация камер")
        sync_button.clicked.connect(on_sync)
        sync_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        distance_model_button = QPushButton("📁 Выбор модели YOLO")
        distance_model_button.clicked.connect(on_select_model)
        distance_model_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        distance_settings_button = QPushButton("⚙️ Настройки")
        distance_settings_button.clicked.connect(on_show_settings)
        distance_settings_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        tools_layout.addWidget(calibration_button)
        tools_layout.addWidget(sync_button)
        tools_layout.addWidget(distance_model_button)
        tools_layout.addWidget(distance_settings_button)
        
        tools_group.setLayout(tools_layout)
        distance_control_layout.addWidget(tools_group)
        
        distance_control_layout.addStretch()
        
        distance_layout.addLayout(distance_video_layout, 7)
        distance_layout.addLayout(distance_control_layout, 3)
        
        # Store all the created widgets in a dictionary for easy access
        widgets = {
            'cam1_combo': cam1_combo,
            'cam2_combo': cam2_combo,
            'active_cam_combo': active_cam_combo,
            'start_distance_button': start_distance_button,
            'stop_distance_button': stop_distance_button,
            'calibration_status_label': calibration_status_label,
            'sync_status_label': sync_status_label,
            'calibration_button': calibration_button,
            'sync_button': sync_button
        }
        
        return distance_layout, widgets
    
    @staticmethod
    def create_mode_selector(on_mode_change):
        """Create mode selector radio buttons."""
        mode_group = QGroupBox("Режим работы")
        mode_layout = QHBoxLayout()
        
        mode_detection = QRadioButton("Распознавание и трекинг")
        mode_detection.setChecked(True)
        mode_detection.toggled.connect(lambda checked: on_mode_change("detection") if checked else None)
        
        mode_distance = QRadioButton("Распознавание, трекинг и измерение расстояния")
        mode_distance.toggled.connect(lambda checked: on_mode_change("distance") if checked else None)
        
        mode_layout.addWidget(mode_detection)
        mode_layout.addWidget(mode_distance)
        mode_layout.addStretch()
        
        mode_group.setLayout(mode_layout)
        
        return mode_group, mode_detection, mode_distance
    
    @staticmethod
    def create_camera_button(camera_name, camera_url, on_select_camera, index=0):
        """Create a camera selection button."""
        btn = QPushButton(f"📹 {camera_name}")
        btn.clicked.connect(lambda: on_select_camera(camera_url))
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setMinimumHeight(40)
        
        # Button colors to cycle through
        button_colors = [
            "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9)",
            "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2ecc71, stop:1 #27ae60)",
            "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b)",
            "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #9b59b6, stop:1 #8e44ad)",
            "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f1c40f, stop:1 #f39c12)"
        ]
        
        color_index = index % len(button_colors)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {button_colors[color_index]};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                    stop:0 #3aaef0, stop:1 #2980b9);
            }}
        """)
        
        return btn 