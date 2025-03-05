from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QComboBox, QCheckBox, QSpinBox,
    QPushButton, QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt

class SettingsDialog(QDialog):
    def __init__(self, parent=None, model_path=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setMinimumWidth(500)
        
        # Установка стилей для диалога
        self.setStyleSheet("""
            QDialog {
                background-color: #F5F6FA;
            }
            
            QGroupBox {
                background-color: white;
                border: 1px solid #E1E1E1;
                border-radius: 8px;
                margin-top: 1em;
                padding: 15px;
                font-weight: bold;
                color: #2C3E50;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #3498DB;
            }
            
            QLabel {
                color: #34495E;
                font-size: 13px;
                background: transparent;
                padding: 2px;
            }
            
            QGroupBox QLabel {
                background-color: transparent;
                color: #34495E;
                font-weight: normal;
            }
            
            QFormLayout QLabel {
                min-width: 150px;
            }
            
            QDoubleSpinBox, QSpinBox {
                padding: 5px;
                border: 1px solid #E1E1E1;
                border-radius: 4px;
                background: white;
                min-width: 100px;
            }
                           
            QComboBox {
                padding: 5px;
                border: 1px solid #E1E1E1;
                border-radius: 4px;
                background: white;
                color: #34495E;
            }
                           
            QComboBox::drop-down {
                border: none;
                width: 50px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #34495E;
                width: 0;
                height: 0;
                margin-right: 5px;
            }

            QComboBox:on {
                border: 1px solid #3498DB;
            }
            
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #E1E1E1;
                selection-background-color: #3498DB;
            }

            QComboBox QAbstractItemView::item {
                color: #34495E; 
                background-color: white;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #3498DB; /* Цвет выделения */
                color: white; /* Цвет текста */
                border: 1px solid #2980B9; /* Обводка для выделенного элемента */
            }
                           
            QComboBox QAbstractItemView::item:hover {
                background-color: #ECF0F1;
                color: #34495E;
            }

            
            QCheckBox {
                color: #34495E;
                background: transparent;
            }

            QCheckBox:hover {
                color: #2980B9;
            }
            
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                min-width: 100px;
            }
            
            QPushButton[text="Сохранить"] {
                background-color: #3498DB;
                color: white;
                border: none;
            }
            
            QPushButton[text="Сохранить"]:hover {
                background-color: #2980B9;
            }
            
            QPushButton[text="Отмена"] {
                background-color: #E1E1E1;
                color: #2C3E50;
                border: none;
            }
            
            QPushButton[text="Отмена"]:hover {
                background-color: #D1D1D1;
            }

            * {
                outline: none;
            }
        """)
        
        # Основной layout
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Информация о модели
        model_group = QGroupBox("Информация о модели")
        model_layout = QFormLayout()
        model_layout.setSpacing(10)
        self.model_label = QLabel(model_path if model_path else "Модель не выбрана")
        model_layout.addRow("Текущая модель:", self.model_label)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Настройки модели YOLO
        yolo_group = QGroupBox("Настройки YOLO")
        yolo_layout = QFormLayout()
        yolo_layout.setSpacing(10)
        
        # Confidence threshold
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        yolo_layout.addRow("Порог уверенности:", self.conf_spin)
        
        # IoU threshold
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        yolo_layout.addRow("IoU порог:", self.iou_spin)
        
        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(['cpu', 'cuda:0', 'mps'])
        yolo_layout.addRow("Устройство:", self.device_combo)
        
        # Half precision
        self.half_check = QCheckBox("Половинная точность (FP16)")
        yolo_layout.addRow("", self.half_check)
        
        yolo_group.setLayout(yolo_layout)
        layout.addWidget(yolo_group)
        
        # Настройки трекера
        tracker_group = QGroupBox("Настройки трекера")
        tracker_layout = QFormLayout()
        tracker_layout.setSpacing(10)
        
        # FPS setting
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        tracker_layout.addRow("Частота кадров (FPS):", self.fps_spin)
        
        tracker_group.setLayout(tracker_layout)
        layout.addWidget(tracker_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        save_button = QPushButton("Сохранить")
        cancel_button = QPushButton("Отмена")
        save_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addStretch()
        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(save_button)
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
    
    def get_settings(self):
        """Возвращает текущие настройки."""
        return {
            'conf': self.conf_spin.value(),
            'iou': self.iou_spin.value(),
            'device': self.device_combo.currentText(),
            'half': self.half_check.isChecked(),
            'fps': self.fps_spin.value()
        }
    
    def set_settings(self, settings):
        """Устанавливает значения настроек."""
        self.conf_spin.setValue(settings.get('conf', 0.25))
        self.iou_spin.setValue(settings.get('iou', 0.45))
        self.device_combo.setCurrentText(settings.get('device', 'cpu'))
        self.half_check.setChecked(settings.get('half', False))
        self.fps_spin.setValue(settings.get('fps', 30))
    
    def update_model_path(self, path):
        """Обновляет отображаемый путь к модели."""
        self.model_label.setText(path if path else "Модель не выбрана") 