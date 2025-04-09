import sys
import cv2
import numpy as np
import time
import json
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QComboBox, QGridLayout, QSpinBox, QDoubleSpinBox, QMessageBox,
    QGroupBox, QApplication, QFileDialog, QCheckBox, QProgressBar
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from camera_utils import convert_cv_qt
from config import Config


class CameraCalibrationThread(QThread):
    """
    Поток для калибровки камер.
    Выполняет калибровку на основе шахматной доски.
    """
    update_signal = Signal(object, str)  # frame, camera_name
    progress_signal = Signal(int)  # progress percentage
    status_signal = Signal(str)  # status message
    finished_signal = Signal(dict)  # calibration results
    
    def __init__(self, camera1_url, camera2_url, chessboard_size=(9, 6), square_size=1.0, num_frames=15):
        super().__init__()
        self.camera1_url = camera1_url
        self.camera2_url = camera2_url
        self.chessboard_size = chessboard_size  # (cols, rows)
        self.square_size = square_size  # cm
        self.num_frames = num_frames
        self.running = False
        
    def stop(self):
        self.running = False
        
    def run(self):
        self.running = True
        self.status_signal.emit("Инициализация камер...")
        
        # Открытие камер
        cap1 = cv2.VideoCapture(self.camera1_url)
        if not cap1.isOpened():
            self.status_signal.emit(f"Ошибка: Не удалось открыть камеру 1 ({self.camera1_url})")
            return
        
        cap2 = cv2.VideoCapture(self.camera2_url)
        if not cap2.isOpened():
            cap1.release()
            self.status_signal.emit(f"Ошибка: Не удалось открыть камеру 2 ({self.camera2_url})")
            return
        
        # Подготовка критериев для калибровки
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Подготовка точек объекта (0,0,0), (1,0,0), (2,0,0) и т.д.
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp = objp * self.square_size  # масштабирование до реальных размеров
        
        # Массивы для хранения точек объекта и изображения
        objpoints1 = []  # 3D точки для камеры 1
        imgpoints1 = []  # 2D точки для камеры 1
        objpoints2 = []  # 3D точки для камеры 2
        imgpoints2 = []  # 2D точки для камеры 2
        
        collected_frames = 0
        
        # Сбор калибровочных данных
        self.status_signal.emit("Сбор калибровочных данных. Держите шахматную доску перед камерами...")
        
        while self.running and collected_frames < self.num_frames:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                self.status_signal.emit("Ошибка: Не удалось получить кадр с одной из камер")
                break
                
            # Преобразование в grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Поиск углов шахматной доски
            ret1, corners1 = cv2.findChessboardCorners(gray1, self.chessboard_size, None)
            ret2, corners2 = cv2.findChessboardCorners(gray2, self.chessboard_size, None)
            
            # Если шахматная доска найдена на обоих изображениях, уточняем углы и сохраняем
            if ret1 and ret2:
                # Уточнение положения углов
                corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                
                # Отображение углов
                frame1_with_corners = frame1.copy()
                frame2_with_corners = frame2.copy()
                cv2.drawChessboardCorners(frame1_with_corners, self.chessboard_size, corners1, ret1)
                cv2.drawChessboardCorners(frame2_with_corners, self.chessboard_size, corners2, ret2)
                
                self.update_signal.emit(frame1_with_corners, "camera1")
                self.update_signal.emit(frame2_with_corners, "camera2")
                
                # Ждем небольшую задержку, чтобы пользователь мог переместить шахматную доску
                time.sleep(1)
                
                # Сохраняем данные
                objpoints1.append(objp)
                imgpoints1.append(corners1)
                objpoints2.append(objp)
                imgpoints2.append(corners2)
                
                collected_frames += 1
                self.progress_signal.emit(int(collected_frames / self.num_frames * 100))
                self.status_signal.emit(f"Собрано {collected_frames}/{self.num_frames} кадров. Переместите шахматную доску.")
            else:
                # Отображаем обычные кадры с камер
                self.update_signal.emit(frame1, "camera1")
                self.update_signal.emit(frame2, "camera2")
            
            # Небольшая задержка
            time.sleep(0.1)
            
        # Освобождение камер
        cap1.release()
        cap2.release()
        
        if not self.running:
            self.status_signal.emit("Калибровка прервана")
            return
            
        if collected_frames < self.num_frames:
            self.status_signal.emit("Недостаточно данных для калибровки")
            return
            
        # Выполнение калибровки
        self.status_signal.emit("Выполнение калибровки камеры 1...")
        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
            objpoints1, imgpoints1, gray1.shape[::-1], None, None
        )
        
        self.status_signal.emit("Выполнение калибровки камеры 2...")
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
            objpoints2, imgpoints2, gray2.shape[::-1], None, None
        )
        
        # Стерео калибровка
        self.status_signal.emit("Выполнение стерео калибровки...")
        calibration_flags = (
            cv2.CALIB_FIX_INTRINSIC
        )
        
        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints1, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2,
            gray1.shape[::-1], criteria=criteria, flags=calibration_flags
        )
        
        # Ректификация
        self.status_signal.emit("Вычисление ректификационных матриц...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx1, dist1, mtx2, dist2, gray1.shape[::-1], R, T
        )
        
        # Вычисление карт ректификации
        self.status_signal.emit("Вычисление карт ректификации...")
        map1x, map1y = cv2.initUndistortRectifyMap(
            mtx1, dist1, R1, P1, gray1.shape[::-1], cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            mtx2, dist2, R2, P2, gray2.shape[::-1], cv2.CV_32FC1
        )
        
        # Подготовка результатов
        calibration_data = {
            "camera1": {
                "matrix": mtx1.tolist(),
                "distortion": dist1.tolist(),
                "R": R1.tolist(),
                "P": P1.tolist(),
                "mapx": map1x.tolist(),
                "mapy": map1y.tolist()
            },
            "camera2": {
                "matrix": mtx2.tolist(),
                "distortion": dist2.tolist(),
                "R": R2.tolist(),
                "P": P2.tolist(),
                "mapx": map2x.tolist(),
                "mapy": map2y.tolist()
            },
            "stereo": {
                "R": R.tolist(),
                "T": T.tolist(),
                "E": E.tolist(),
                "F": F.tolist(),
                "Q": Q.tolist()
            },
            "info": {
                "camera1_url": self.camera1_url,
                "camera2_url": self.camera2_url,
                "chessboard_size": self.chessboard_size,
                "square_size": self.square_size,
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        self.status_signal.emit("Калибровка завершена успешно!")
        self.finished_signal.emit(calibration_data)


class CalibrationDialog(QDialog):
    """
    Диалоговое окно для калибровки камер.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.config = Config()
        
        self.setWindowTitle("Калибровка камер")
        self.resize(1200, 800)
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
                background-color: #3498DB;
                color: white;
                border: 1px solid #2980B9;
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
        
        # Инициализация интерфейса
        self.init_ui()
        
        # Загрузка камер
        self.load_cameras()
        
        # Переменные для калибровки
        self.calibration_thread = None
        self.calibration_data = None
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        main_layout = QVBoxLayout()
        
        # Выбор камер
        cameras_group = QGroupBox("Выбор камер")
        cameras_layout = QGridLayout()
        
        cameras_layout.addWidget(QLabel("Камера 1:"), 0, 0)
        self.camera1_combo = QComboBox()
        cameras_layout.addWidget(self.camera1_combo, 0, 1)
        
        cameras_layout.addWidget(QLabel("Камера 2:"), 1, 0)
        self.camera2_combo = QComboBox()
        cameras_layout.addWidget(self.camera2_combo, 1, 1)
        
        # Параметры калибровки
        cameras_layout.addWidget(QLabel("Размер шахматной доски:"), 2, 0)
        self.chessboard_cols = QSpinBox()
        self.chessboard_cols.setMinimum(3)
        self.chessboard_cols.setMaximum(20)
        self.chessboard_cols.setValue(9)
        cameras_layout.addWidget(self.chessboard_cols, 2, 1)
        cameras_layout.addWidget(QLabel("x"), 2, 2)
        self.chessboard_rows = QSpinBox()
        self.chessboard_rows.setMinimum(3)
        self.chessboard_rows.setMaximum(20)
        self.chessboard_rows.setValue(6)
        cameras_layout.addWidget(self.chessboard_rows, 2, 3)
        
        cameras_layout.addWidget(QLabel("Размер квадрата (см):"), 3, 0)
        self.square_size = QDoubleSpinBox()
        self.square_size.setDecimals(2)
        self.square_size.setMinimum(0.1)
        self.square_size.setMaximum(10.0)
        self.square_size.setValue(1.0)
        cameras_layout.addWidget(self.square_size, 3, 1)
        
        cameras_layout.addWidget(QLabel("Количество кадров:"), 4, 0)
        self.num_frames = QSpinBox()
        self.num_frames.setMinimum(5)
        self.num_frames.setMaximum(50)
        self.num_frames.setValue(15)
        cameras_layout.addWidget(self.num_frames, 4, 1)
        
        cameras_group.setLayout(cameras_layout)
        main_layout.addWidget(cameras_group)
        
        # Окна просмотра видео
        video_layout = QHBoxLayout()
        
        self.video_label1 = QLabel("Камера 1")
        self.video_label1.setMinimumSize(480, 360)
        self.video_label1.setAlignment(Qt.AlignCenter)
        self.video_label1.setStyleSheet("background-color: black; color: white;")
        video_layout.addWidget(self.video_label1)
        
        self.video_label2 = QLabel("Камера 2")
        self.video_label2.setMinimumSize(480, 360)
        self.video_label2.setAlignment(Qt.AlignCenter)
        self.video_label2.setStyleSheet("background-color: black; color: white;")
        video_layout.addWidget(self.video_label2)
        
        main_layout.addLayout(video_layout)
        
        # Прогресс и статус
        progress_layout = QVBoxLayout()
        self.status_label = QLabel("Готово к калибровке")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(progress_layout)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Начать калибровку")
        self.start_button.clicked.connect(self.start_calibration)
        buttons_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Остановить")
        self.stop_button.clicked.connect(self.stop_calibration)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        self.save_button = QPushButton("Сохранить результаты")
        self.save_button.clicked.connect(self.save_calibration)
        self.save_button.setEnabled(False)
        buttons_layout.addWidget(self.save_button)
        
        main_layout.addLayout(buttons_layout)
        
        self.setLayout(main_layout)
    
    def load_cameras(self):
        """Загружает список камер из файла cameras.txt."""
        try:
            with open("cameras.txt", "r") as f:
                camera_lines = [line.strip() for line in f if line.strip()]
            
            self.camera1_combo.clear()
            self.camera2_combo.clear()
            
            # Обработка строк из cameras.txt
            cameras = []
            camera_names = []
            
            for line in camera_lines:
                parts = line.strip().split(' ', 1)  # Разделяем на имя и URL
                if len(parts) == 2:
                    camera_name = parts[0]
                    camera_url = parts[1]
                    cameras.append(camera_url)
                    camera_names.append(camera_name)
                else:
                    # Если формат неверный, используем всю строку как URL
                    cameras.append(line)
                    camera_names.append(f"Камера {len(cameras)}")
            
            for i, (camera_url, camera_name) in enumerate(zip(cameras, camera_names)):
                self.camera1_combo.addItem(f"{camera_name}", camera_url)
                self.camera2_combo.addItem(f"{camera_name}", camera_url)
            
            # Устанавливаем разные камеры по умолчанию, если их более одной
            if len(cameras) > 1:
                self.camera2_combo.setCurrentIndex(1)
                
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Файл cameras.txt не найден. Пожалуйста, создайте файл со списком камер."
            )
    
    def update_frame(self, frame, camera_name):
        """Обновляет кадр из указанной камеры."""
        if frame is not None:
            pixmap = QPixmap.fromImage(convert_cv_qt(frame))
            if camera_name == "camera1":
                self.video_label1.setPixmap(pixmap)
            elif camera_name == "camera2":
                self.video_label2.setPixmap(pixmap)
    
    def update_progress(self, value):
        """Обновляет прогресс-бар."""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """Обновляет статусную строку."""
        self.status_label.setText(message)
    
    def calibration_finished(self, calibration_data):
        """Обрабатывает завершение калибровки."""
        self.calibration_data = calibration_data
        self.save_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def start_calibration(self):
        """Запускает процесс калибровки камер."""
        if self.camera1_combo.currentText() == self.camera2_combo.currentText():
            QMessageBox.warning(self, "Ошибка", "Выберите разные камеры для калибровки")
            return
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Получаем параметры калибровки
        camera1_url = self.camera1_combo.currentData()
        camera2_url = self.camera2_combo.currentData()
        chessboard_size = (self.chessboard_cols.value(), self.chessboard_rows.value())
        square_size = self.square_size.value()
        num_frames = self.num_frames.value()
        
        # Запускаем поток калибровки
        self.calibration_thread = CameraCalibrationThread(
            camera1_url, camera2_url, chessboard_size, square_size, num_frames
        )
        self.calibration_thread.update_signal.connect(self.update_frame)
        self.calibration_thread.progress_signal.connect(self.update_progress)
        self.calibration_thread.status_signal.connect(self.update_status)
        self.calibration_thread.finished_signal.connect(self.calibration_finished)
        self.calibration_thread.start()
    
    def stop_calibration(self):
        """Останавливает процесс калибровки."""
        if self.calibration_thread and self.calibration_thread.isRunning():
            self.calibration_thread.stop()
            self.status_label.setText("Калибровка остановлена")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def save_calibration(self):
        """Сохраняет результаты калибровки."""
        if not self.calibration_data:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return
        
        try:
            # Сохранение в файл
            with open("calibration_data.json", "w") as f:
                json.dump(self.calibration_data, f, indent=2)
            
            # Обновление статуса калибровки в Config
            camera1_url = self.camera1_combo.currentData()
            camera2_url = self.camera2_combo.currentData()
            self.config.update_calibration_status(
                True, [camera1_url, camera2_url]
            )
            
            QMessageBox.information(
                self,
                "Успех",
                "Данные калибровки успешно сохранены в файл calibration_data.json"
            )
            
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось сохранить данные калибровки: {str(e)}"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = CalibrationDialog()
    dialog.show()
    sys.exit(app.exec()) 