from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QHBoxLayout, QFormLayout,
    QCheckBox, QLabel, QDoubleSpinBox, QPushButton, QMessageBox
)
from PySide6.QtCore import Qt
from config import Config  # Добавляем импорт Config

class DistanceMeasureDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Создаем экземпляр Config для работы с настройками
        self.config = Config()
        
        # Загружаем текущие настройки
        current_settings = self.config.get_distance_measure_settings()

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
        self.setWindowTitle("Измерение расстояния")
        self.setMinimumWidth(400)
        self.selected_cameras = []
        self.baseline = 0.0

        # Переключатель включения модуля
        self.enabled_checkbox = QCheckBox("Включить измерение расстояния")
        self.enabled_checkbox.setChecked(current_settings.get('enabled', False))
        self.enabled_checkbox.toggled.connect(self.on_enabled_toggled)

        # Основной layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.enabled_checkbox)

        # Группа для списка камер
        self.cameras_group = QGroupBox("Выберите камеры (ровно 2 шт.)")
        cameras_layout = QVBoxLayout()
        self.checkboxes = []
        try:
            with open("cameras.txt", "r") as f:
                saved_cams = current_settings.get('cameras', [])
                for line in f:
                    cam = line.strip()
                    if cam:
                        cb = QCheckBox(cam)
                        # Если камера ранее была выбрана - установить флажок
                        if cam in saved_cams:
                            cb.setChecked(True)
                        cb.toggled.connect(self.on_checkbox_toggled)
                        self.checkboxes.append(cb)
                        cameras_layout.addWidget(cb)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка чтения файла cameras.txt: {e}")

        self.cameras_group.setLayout(cameras_layout)
        main_layout.addWidget(self.cameras_group)

        # Поле для ввода базиса (float)
        form_layout = QFormLayout()
        self.baseline_spin = QDoubleSpinBox()
        self.baseline_spin.setDecimals(2)
        self.baseline_spin.setRange(0.0, 10000.0)
        self.baseline_spin.setValue(current_settings.get('baseline', 10.0))
        form_layout.addRow("Базис (см):", self.baseline_spin)
        main_layout.addLayout(form_layout)

        # Кнопки
        buttons_layout = QHBoxLayout()
        save_button = QPushButton("Сохранить")
        cancel_button = QPushButton("Отмена")
        save_button.clicked.connect(self.on_save)
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addStretch()
        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(save_button)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

        # Изначально отключаем группы, если модуль не включен
        self.on_enabled_toggled(self.enabled_checkbox.isChecked())

    def on_enabled_toggled(self, checked):
        # Включаем или отключаем выбор камер и базиса
        self.cameras_group.setEnabled(checked)
        self.baseline_spin.setEnabled(checked)

    def on_checkbox_toggled(self, checked):
        # Гарантируем, что выбрано не более двух камер
        selected = [cb for cb in self.checkboxes if cb.isChecked()]
        if len(selected) > 2:
            sender = self.sender()
            QMessageBox.warning(self, "Ошибка", "Можно выбрать не более 2-х камер!")
            sender.setChecked(False)

    def on_save(self):
        if self.enabled_checkbox.isChecked():
            selected = [cb.text() for cb in self.checkboxes if cb.isChecked()]
            if len(selected) != 2:
                QMessageBox.warning(self, "Ошибка", "Выберите ровно 2 камеры для измерения расстояния!")
                return
            self.selected_cameras = selected
            self.baseline = self.baseline_spin.value()
        else:
            self.selected_cameras = []
            self.baseline = self.baseline_spin.value()

        # Сохраняем настройки в settings.json
        self.config.update_distance_measure_settings(
            self.enabled_checkbox.isChecked(),
            self.baseline,
            self.selected_cameras
        )
        self.accept()

    def get_values(self):
        return self.enabled_checkbox.isChecked(), self.selected_cameras, self.baseline 