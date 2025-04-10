import cv2
import numpy as np
import os
import json
import time
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
    QMessageBox, QTabWidget, QWidget, QTextEdit, QCheckBox
)
from PySide6.QtCore import Qt, Signal, QThread, QMutex, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO
import math

class DistanceCalculationThread(QThread):
    frame_signal = Signal(object, object, dict)  # Кадр, обработанный кадр с расстояниями, метаданные
    error_signal = Signal(str)
    
    def __init__(self, camera1_url, camera2_url, model_path, baseline, calibration_data=None, sync_data=None):
        super().__init__()
        self.camera1_url = camera1_url
        self.camera2_url = camera2_url
        self.model_path = model_path
        self.baseline = baseline  # в сантиметрах
        self.calibration_data = calibration_data
        self.sync_data = sync_data
        self.running = False
        self.lock = QMutex()
        self.detected_objects = {}
        
    def run(self):
        self.running = True
        
        # Загружаем YOLO модель
        try:
            model = YOLO(self.model_path)
        except Exception as e:
            self.error_signal.emit(f"Ошибка загрузки модели: {e}")
            self.running = False
            return
        
        # Открываем видеопотоки
        cap1 = cv2.VideoCapture(self.camera1_url)
        cap2 = cv2.VideoCapture(self.camera2_url)
        
        if not cap1.isOpened() or not cap2.isOpened():
            self.error_signal.emit("Не удалось открыть одну или обе камеры")
            self.running = False
            return
        
        # Загружаем матрицы калибровки, если есть
        camera_matrix1 = None
        camera_matrix2 = None
        dist_coeffs1 = None
        dist_coeffs2 = None
        
        if self.calibration_data:
            # Проверяем формат данных калибровки
            if 'camera1' in self.calibration_data and 'camera2' in self.calibration_data:
                # Используем данные калибровки из структуры с camera1 и camera2
                try:
                    camera_matrix1 = np.array(self.calibration_data['camera1']['matrix'])
                    dist_coeffs1 = np.array(self.calibration_data['camera1']['distortion'])
                    
                    camera_matrix2 = np.array(self.calibration_data['camera2']['matrix'])
                    dist_coeffs2 = np.array(self.calibration_data['camera2']['distortion'])
                except (KeyError, TypeError) as e:
                    print(f"Ошибка доступа к данным калибровки: {e}")
            # Проверяем старый формат по URL
            elif self.camera1_url in self.calibration_data and self.camera2_url in self.calibration_data:
                camera_matrix1 = np.array(self.calibration_data[self.camera1_url]['camera_matrix'])
                dist_coeffs1 = np.array(self.calibration_data[self.camera1_url]['dist_coeffs'])
                
                camera_matrix2 = np.array(self.calibration_data[self.camera2_url]['camera_matrix'])
                dist_coeffs2 = np.array(self.calibration_data[self.camera2_url]['dist_coeffs'])
            # Если ничего подходящего не найдено, выводим информацию
            else:
                print("Данные калибровки имеются, но не соответствуют ожидаемому формату")
        
        drift_rate = 0
        if self.sync_data:
            drift_rate = self.sync_data.get('drift_rate', 0)
        
        frame_count = 0
        start_time = time.time()
        
        # Основной цикл обработки
        while self.running:
            # Захват кадров
            ret1, frame1 = cap1.read()
            if not ret1:
                continue
                
            # Компенсация расхождения камер
            frames_to_skip = int(drift_rate * (time.time() - start_time))
            if frames_to_skip > 0:
                for _ in range(frames_to_skip):
                    cap2.read()  # Пропускаем кадры для синхронизации
            
            ret2, frame2 = cap2.read()
            if not ret2:
                continue
            
            # Коррекция искажений, если есть данные калибровки
            if camera_matrix1 is not None and dist_coeffs1 is not None:
                try:
                    h, w = frame1.shape[:2]
                    newcameramtx1, roi1 = cv2.getOptimalNewCameraMatrix(camera_matrix1, dist_coeffs1, (w,h), 0, (w,h))
                    frame1 = cv2.undistort(frame1, camera_matrix1, dist_coeffs1, None, newcameramtx1)
                except Exception as e:
                    print(f"Ошибка коррекции искажений камеры 1: {e}")
                
            if camera_matrix2 is not None and dist_coeffs2 is not None:
                try:
                    h, w = frame2.shape[:2]
                    newcameramtx2, roi2 = cv2.getOptimalNewCameraMatrix(camera_matrix2, dist_coeffs2, (w,h), 0, (w,h))
                    frame2 = cv2.undistort(frame2, camera_matrix2, dist_coeffs2, None, newcameramtx2)
                except Exception as e:
                    print(f"Ошибка коррекции искажений камеры 2: {e}")
            
            # Распознавание объектов на обоих кадрах
            results1 = model(frame1)
            results2 = model(frame2)
            
            # Создаем копии для отображения
            display_frame1 = frame1.copy()
            display_frame2 = frame2.copy()
            
            # Добавляем маркеры на кадры, чтобы их можно было отличить
            cv2.putText(display_frame1, "CAM 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(display_frame2, "CAM 2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Обрабатываем результаты
            detections = {}
            
            # Создаем словари для хранения нарисованных объектов на каждой камере
            objects_cam1 = {}
            objects_cam2 = {}
            
            # Список всех обнаруженных объектов для поиска соответствий
            all_detected_objects = []
            
            # Обрабатываем объекты с первой камеры
            if len(results1) > 0:
                boxes1 = results1[0].boxes
                for i, box in enumerate(boxes1):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf)
                    cls_id = int(box.cls)
                    cls_name = model.names[cls_id]
                    
                    center_x1 = (x1 + x2) // 2
                    center_y1 = (y1 + y2) // 2
                    
                    # Добавляем в список для последующего поиска соответствий
                    all_detected_objects.append({
                        'camera': 1,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'center_x': center_x1, 'center_y': center_y1,
                        'cls_id': cls_id, 'cls_name': cls_name, 'conf': conf,
                        'box_index': i
                    })
                    
                    # Временный идентификатор объекта
                    obj_id = f"{cls_name}_{i}"
                    objects_cam1[obj_id] = {
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x1, center_y1),
                        'class': cls_name,
                        'class_id': cls_id,
                        'confidence': conf
                    }
            
            # Обрабатываем объекты со второй камеры
            if len(results2) > 0:
                boxes2 = results2[0].boxes
                for i, box in enumerate(boxes2):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf)
                    cls_id = int(box.cls)
                    cls_name = model.names[cls_id]
                    
                    center_x2 = (x1 + x2) // 2
                    center_y2 = (y1 + y2) // 2
                    
                    # Добавляем в список для последующего поиска соответствий
                    all_detected_objects.append({
                        'camera': 2,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'center_x': center_x2, 'center_y': center_y2,
                        'cls_id': cls_id, 'cls_name': cls_name, 'conf': conf,
                        'box_index': i
                    })
                    
                    # Временный идентификатор объекта
                    obj_id = f"{cls_name}_{i}"
                    objects_cam2[obj_id] = {
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x2, center_y2),
                        'class': cls_name,
                        'class_id': cls_id,
                        'confidence': conf
                    }
            
            # Находим соответствия между объектами и вычисляем расстояния
            matched_pairs = []
            
            for obj1 in [obj for obj in all_detected_objects if obj['camera'] == 1]:
                best_match = None
                min_distance = float('inf')
                
                for obj2 in [obj for obj in all_detected_objects if obj['camera'] == 2]:
                    # Проверяем, что это тот же класс объекта
                    if obj1['cls_id'] == obj2['cls_id']:
                        # Диспаритет - это разница в x-координатах центра объекта
                        disparity = abs(obj1['center_x'] - obj2['center_x'])
                        
                        # Вычисляем расстояние по формуле: distance = (baseline * focal_length) / disparity
                        if disparity > 0:
                            # Используем фокусное расстояние из калибровки или приблизительное значение
                            focal_length = 800  # примерное значение
                            if camera_matrix1 is not None:
                                focal_length = camera_matrix1[0, 0]
                                
                            # Расстояние в сантиметрах
                            distance = (self.baseline * focal_length) / disparity
                            
                            # Находим лучшее соответствие по минимальному расстоянию
                            if distance < min_distance:
                                min_distance = distance
                                best_match = obj2
                                best_match['distance'] = distance
                
                # Если найдено соответствие, сохраняем пару и рисуем на обоих кадрах
                if best_match:
                    matched_pairs.append((obj1, best_match))
                    
                    distance = best_match['distance']
                    
                    # Цвет зависит от расстояния
                    if distance < 200:  # ближе 2 метров
                        color = (0, 0, 255)  # красный для близких объектов
                    elif distance < 500:  # 2-5 метров
                        color = (0, 255, 255)  # желтый для среднего расстояния
                    else:
                        color = (0, 255, 0)  # зеленый для далеких объектов
                    
                    # Рисуем на первом кадре
                    cv2.rectangle(display_frame1, 
                                 (obj1['x1'], obj1['y1']), 
                                 (obj1['x2'], obj1['y2']), 
                                 color, 2)
                    
                    # Текст с информацией на первом кадре
                    label1 = f"{obj1['cls_name']} {distance/100:.2f}m"
                    cv2.putText(display_frame1, label1, 
                               (obj1['x1'], obj1['y1'] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Рисуем на втором кадре
                    cv2.rectangle(display_frame2, 
                                 (best_match['x1'], best_match['y1']), 
                                 (best_match['x2'], best_match['y2']), 
                                 color, 2)
                    
                    # Текст с информацией на втором кадре
                    label2 = f"{best_match['cls_name']} {distance/100:.2f}m"
                    cv2.putText(display_frame2, label2, 
                               (best_match['x1'], best_match['y1'] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Добавляем в общий список детекций для интерфейса
                    obj_id = f"{obj1['cls_name']}_{obj1['box_index']}"
                    detections[obj_id] = {
                        'class': obj1['cls_name'],
                        'distance': distance/100,  # в метрах
                        'position_cam1': (obj1['center_x'], obj1['center_y']),
                        'position_cam2': (best_match['center_x'], best_match['center_y']),
                        'bbox_cam1': (obj1['x1'], obj1['y1'], obj1['x2'], obj1['y2']),
                        'bbox_cam2': (best_match['x1'], best_match['y1'], best_match['x2'], best_match['y2']),
                        'confidence': obj1['conf'] * best_match['conf']  # комбинированная уверенность
                    }
            
            # Рисуем непарные объекты на первой камере
            for obj_id, obj_data in objects_cam1.items():
                # Пропускаем объекты, которые уже обработаны в парах
                if any(matched_obj1['box_index'] == int(obj_id.split('_')[1]) and 
                       matched_obj1['cls_name'] == obj_data['class'] 
                       for matched_obj1, _ in matched_pairs):
                    continue
                
                # Рисуем объект без информации о расстоянии
                x1, y1, x2, y2 = obj_data['bbox']
                cls_name = obj_data['class']
                
                # Серый цвет для объектов без пары
                color = (128, 128, 128)
                
                cv2.rectangle(display_frame1, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} (no match)"
                cv2.putText(display_frame1, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Рисуем непарные объекты на второй камере
            for obj_id, obj_data in objects_cam2.items():
                # Пропускаем объекты, которые уже обработаны в парах
                if any(matched_obj2['box_index'] == int(obj_id.split('_')[1]) and 
                       matched_obj2['cls_name'] == obj_data['class'] 
                       for _, matched_obj2 in matched_pairs):
                    continue
                
                # Рисуем объект без информации о расстоянии
                x1, y1, x2, y2 = obj_data['bbox']
                cls_name = obj_data['class']
                
                # Серый цвет для объектов без пары
                color = (128, 128, 128)
                
                cv2.rectangle(display_frame2, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} (no match)"
                cv2.putText(display_frame2, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Добавляем информацию о кадре
            frame_info = {
                'frame_count': frame_count,
                'timestamp': time.time() - start_time,
                'num_detections': len(detections),
                'detections': detections
            }
            
            # Отправляем данные в основной поток - оба обработанных кадра
            self.frame_signal.emit(display_frame1, display_frame2, frame_info)
            
            frame_count += 1
            
        # Освобождаем ресурсы
        cap1.release()
        cap2.release()
        
    def stop(self):
        self.running = False
        self.wait()

class DistanceCalculatorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Измерение расстояния")
        self.setMinimumSize(1200, 800)
        
        self.calibration_data = {}
        self.sync_data = {}
        self.calculation_thread = None
        
        # Проверяем доступность необходимых данных
        self.is_calibrated = False
        self.is_synced = False
        self.load_calibration_data()
        self.load_sync_data()
        
        # Основной layout
        main_layout = QVBoxLayout()
        
        # Верхняя панель с информацией о статусе
        status_layout = QHBoxLayout()
        
        self.calibration_status_label = QLabel(f"Калибровка: {'✅' if self.is_calibrated else '❌'}")
        self.sync_status_label = QLabel(f"Синхронизация: {'✅' if self.is_synced else '❌'}")
        
        status_layout.addWidget(self.calibration_status_label)
        status_layout.addWidget(self.sync_status_label)
        
        # Добавляем кнопки для выполнения калибровки и синхронизации, если нужно
        if not self.is_calibrated:
            calibrate_btn = QPushButton("Выполнить калибровку")
            calibrate_btn.clicked.connect(self.on_calibrate_clicked)
            status_layout.addWidget(calibrate_btn)
            
        if not self.is_synced:
            sync_btn = QPushButton("Выполнить синхронизацию")
            sync_btn.clicked.connect(self.on_sync_clicked)
            status_layout.addWidget(sync_btn)
        
        main_layout.addLayout(status_layout)
        
        # Панель настроек
        settings_group = QGroupBox("Настройки измерения расстояния")
        settings_layout = QGridLayout()
        
        # Выбор камер
        settings_layout.addWidget(QLabel("Камера 1:"), 0, 0)
        self.cam1_combo = QComboBox()
        settings_layout.addWidget(self.cam1_combo, 0, 1)
        
        settings_layout.addWidget(QLabel("Камера 2:"), 1, 0)
        self.cam2_combo = QComboBox()
        settings_layout.addWidget(self.cam2_combo, 1, 1)
        
        # Загружаем список камер и устанавливаем откалиброванные
        self.load_cameras()
        
        # Выбор модели YOLO
        settings_layout.addWidget(QLabel("Модель YOLO:"), 0, 2)
        self.model_combo = QComboBox()
        self.load_models()
        settings_layout.addWidget(self.model_combo, 0, 3)
        
        # Базис (расстояние между камерами)
        settings_layout.addWidget(QLabel("Базис (см):"), 1, 2)
        self.baseline_spin = QDoubleSpinBox()
        self.baseline_spin.setRange(1, 1000)
        self.baseline_spin.setValue(10.0)
        self.baseline_spin.setDecimals(1)
        settings_layout.addWidget(self.baseline_spin, 1, 3)
        
        # Загружаем базис из настроек, если есть
        if self.parent() and hasattr(self.parent(), 'config'):
            distance_settings = self.parent().config.get_distance_measure_settings()
            if 'baseline' in distance_settings:
                self.baseline_spin.setValue(distance_settings['baseline'])
        
        # Режим отображения
        settings_layout.addWidget(QLabel("Показывать для камеры:"), 0, 4)
        self.display_combo = QComboBox()
        self.display_combo.addItem("Камера 1")
        self.display_combo.addItem("Камера 2")
        settings_layout.addWidget(self.display_combo, 0, 5)
        
        # Показывать расстояния
        self.show_distances_check = QCheckBox("Показывать расстояния")
        self.show_distances_check.setChecked(True)
        settings_layout.addWidget(self.show_distances_check, 1, 4, 1, 2)
        
        # Кнопки управления
        self.start_btn = QPushButton("Запустить измерение")
        self.start_btn.clicked.connect(self.start_distance_calculation)
        settings_layout.addWidget(self.start_btn, 0, 6, 2, 1)
        
        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.clicked.connect(self.stop_distance_calculation)
        self.stop_btn.setEnabled(False)
        settings_layout.addWidget(self.stop_btn, 0, 7, 2, 1)
        
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # Область отображения
        display_layout = QHBoxLayout()
        
        # Видео с камеры
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setText("Нажмите 'Запустить измерение' для начала работы")
        display_layout.addWidget(self.video_label, 7)
        
        # Информация о детекции
        info_layout = QVBoxLayout()
        
        # Информация о распознанных объектах
        objects_group = QGroupBox("Распознанные объекты")
        objects_layout = QVBoxLayout()
        self.objects_text = QTextEdit()
        self.objects_text.setReadOnly(True)
        objects_layout.addWidget(self.objects_text)
        objects_group.setLayout(objects_layout)
        info_layout.addWidget(objects_group)
        
        # Статистика
        stats_group = QGroupBox("Статистика")
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        info_layout.addWidget(stats_group)
        
        display_layout.addLayout(info_layout, 3)
        
        main_layout.addLayout(display_layout)
        
        self.setLayout(main_layout)
        
        # Блокируем кнопку старта, если нет калибровки или синхронизации
        self.start_btn.setEnabled(self.is_calibrated and self.is_synced)
        if not self.start_btn.isEnabled():
            warning = "⚠️ Для измерения расстояния необходимо выполнить калибровку и синхронизацию камер"
            self.stats_text.append(warning)
        
        # Таймер для обновления информации
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_statistics)
        self.update_timer.start(1000)  # Обновляем статистику каждую секунду
        
    def load_cameras(self):
        try:
            with open("cameras.txt", "r") as f:
                camera_lines = [line.strip() for line in f if line.strip()]
                
            self.cam1_combo.clear()
            self.cam2_combo.clear()
            
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
                self.cam1_combo.addItem(f"{camera_name}", camera_url)
                self.cam2_combo.addItem(f"{camera_name}", camera_url)
                
            if len(cameras) > 1:
                self.cam2_combo.setCurrentIndex(1)
                    
            # Если есть ранее выбранные камеры в настройках, устанавливаем их
            if self.parent() and hasattr(self.parent(), 'config'):
                distance_settings = self.parent().config.get_distance_measure_settings()
                if 'cameras' in distance_settings and len(distance_settings['cameras']) >= 2:
                    cam1 = distance_settings['cameras'][0]
                    cam2 = distance_settings['cameras'][1]
                    
                    # Находим сохраненные камеры по URL в данных комбобоксов
                    for i in range(self.cam1_combo.count()):
                        if self.cam1_combo.itemData(i) == cam1:
                            self.cam1_combo.setCurrentIndex(i)
                            break
                            
                    for i in range(self.cam2_combo.count()):
                        if self.cam2_combo.itemData(i) == cam2:
                            self.cam2_combo.setCurrentIndex(i)
                            break
                        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка чтения файла cameras.txt: {e}")
            
    def load_models(self):
        # Поиск .pt файлов в каталоге models
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
            if model_files:
                self.model_combo.addItems([os.path.join(models_dir, f) for f in model_files])
        
        # Если есть файлы .pt в корневом каталоге, добавляем их тоже
        root_models = [f for f in os.listdir() if f.endswith('.pt')]
        if root_models:
            self.model_combo.addItems(root_models)
            
        # Проверяем настройки, если есть модель по умолчанию
        if self.parent() and hasattr(self.parent(), 'config'):
            # Используем get_last_model вместо get_model_settings
            last_model = self.parent().config.get_last_model()
            if last_model:
                index = self.model_combo.findText(last_model)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
                else:
                    # Если модель не найдена в списке, добавляем её
                    self.model_combo.addItem(last_model)
                    self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
    
    def load_calibration_data(self):
        """Загружает данные калибровки из файла calibration_data.json."""
        calibration_file = "calibration_data.json"
        if os.path.exists(calibration_file):
            try:
                with open(calibration_file, "r") as f:
                    self.calibration_data = json.load(f)
                if self.calibration_data:
                    self.is_calibrated = True
                    print(f"Загружены данные калибровки из {calibration_file}")
            except Exception as e:
                print(f"Ошибка при загрузке данных калибровки: {e}")
        else:
            # Пробуем альтернативные пути для поиска файла калибровки
            alt_paths = [
                os.path.join("calibration", "camera_calibration.json"),
                os.path.join("calibration", "calibration_data.json")
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    try:
                        with open(path, "r") as f:
                            self.calibration_data = json.load(f)
                        if self.calibration_data:
                            self.is_calibrated = True
                            print(f"Загружены данные калибровки из {path}")
                            break
                    except Exception as e:
                        print(f"Ошибка при загрузке данных калибровки из {path}: {e}")
            
            if not self.is_calibrated:
                print("Файл калибровки не найден")
    
    def load_sync_data(self):
        """Загружает данные синхронизации из файла sync_data.json."""
        sync_file = "sync_data.json"
        if os.path.exists(sync_file):
            try:
                with open(sync_file, "r") as f:
                    self.sync_data = json.load(f)
                if self.sync_data and 'time_diff' in self.sync_data:
                    self.is_synced = True
                    print(f"Загружены данные синхронизации из {sync_file}")
            except Exception as e:
                print(f"Ошибка при загрузке данных синхронизации: {e}")
        else:
            # Пробуем найти файлы в каталоге sync
            sync_dir = "sync"
            if os.path.exists(sync_dir):
                # Ищем последний файл синхронизации
                sync_files = [f for f in os.listdir(sync_dir) if f.startswith("sync_") and f.endswith(".json")]
                if sync_files:
                    latest_file = max(sync_files, key=lambda f: os.path.getmtime(os.path.join(sync_dir, f)))
                    
                    try:
                        with open(os.path.join(sync_dir, latest_file), "r") as f:
                            self.sync_data = json.load(f)
                        if self.sync_data and 'time_diff' in self.sync_data:
                            self.is_synced = True
                            print(f"Загружены данные синхронизации из {os.path.join(sync_dir, latest_file)}")
                    except Exception as e:
                        print(f"Ошибка при загрузке данных синхронизации: {e}")
            
            if not self.is_synced:
                print("Файл синхронизации не найден")
    
    def on_calibrate_clicked(self):
        # Вызов диалога калибровки
        if self.parent():
            self.parent().open_calibration_dialog()
            
    def on_sync_clicked(self):
        # Вызов диалога синхронизации
        if self.parent():
            self.parent().open_sync_dialog()
    
    def start_distance_calculation(self):
        # Проверяем, выбраны ли разные камеры
        cam1_name = self.cam1_combo.currentText()
        cam2_name = self.cam2_combo.currentText()
        cam1_url = self.cam1_combo.currentData()
        cam2_url = self.cam2_combo.currentData()
        
        if cam1_name == cam2_name:
            QMessageBox.warning(self, "Предупреждение", "Выберите разные камеры для измерения расстояния")
            return
            
        # Проверяем, выбрана ли модель
        model_path = self.model_combo.currentText()
        if not model_path:
            QMessageBox.warning(self, "Предупреждение", "Выберите модель YOLO")
            return
            
        baseline = self.baseline_spin.value()
        
        # Создаем и запускаем поток расчета расстояния
        self.calculation_thread = DistanceCalculationThread(
            cam1_url, cam2_url, model_path, baseline, 
            self.calibration_data, self.sync_data
        )
        self.calculation_thread.frame_signal.connect(self.update_display)
        self.calculation_thread.error_signal.connect(self.on_error)
        
        self.calculation_thread.start()
        
        # Обновляем UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.cam1_combo.setEnabled(False)
        self.cam2_combo.setEnabled(False)
        self.model_combo.setEnabled(False)
        
        # Очищаем текстовые поля
        self.objects_text.clear()
        self.stats_text.clear()
        self.stats_text.append("📊 Статистика обработки:")
        self.stats_text.append(f"Камера 1: {cam1_name} ({cam1_url})")
        self.stats_text.append(f"Камера 2: {cam2_name} ({cam2_url})")
        self.stats_text.append(f"Модель: {model_path}")
        self.stats_text.append(f"Базис: {baseline} см")
        if self.is_calibrated:
            self.stats_text.append("✅ Камеры откалиброваны")
        if self.is_synced:
            self.stats_text.append(f"✅ Камеры синхронизированы (drift_rate: {self.sync_data.get('drift_rate', 0):.2f})")
    
    def stop_distance_calculation(self):
        if self.calculation_thread and self.calculation_thread.isRunning():
            self.calculation_thread.stop()
            
        # Обновляем UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.cam1_combo.setEnabled(True)
        self.cam2_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        
        # Очищаем дисплей
        self.video_label.setText("Нажмите 'Запустить измерение' для начала работы")
    
    def update_display(self, original_frame, processed_frame, info):
        # Определяем, какую камеру показывать
        display_index = self.display_combo.currentIndex()
        frame_to_display = processed_frame if display_index == 0 else original_frame
        
        # Конвертируем кадр для отображения
        rgb_image = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Обновляем информацию о распознанных объектах
        self.objects_text.clear()
        detections = info.get('detections', {})
        if detections:
            for obj_id, obj_data in detections.items():
                cls_name = obj_data['class']
                distance = obj_data['distance']
                confidence = obj_data['confidence']
                
                # Разный цвет для разных расстояний
                if distance < 2:  # ближе 2 метров
                    color = "red"
                elif distance < 5:  # 2-5 метров
                    color = "orange"
                else:
                    color = "green"
                    
                self.objects_text.append(
                    f"<span style='color:{color};'><b>{cls_name}</b>: "
                    f"{distance:.2f} м (уверенность: {confidence:.2f})</span>"
                )
        else:
            self.objects_text.append("Объекты не обнаружены")
    
    def update_statistics(self):
        if self.calculation_thread and self.calculation_thread.isRunning():
            # Здесь можно добавить обновление статистики обработки, 
            # например FPS, если нужно
            pass
    
    def on_error(self, error_msg):
        QMessageBox.critical(self, "Ошибка", error_msg)
        self.stop_distance_calculation()
    
    def closeEvent(self, event):
        self.stop_distance_calculation()
        event.accept()

    def start_automatic_measurement(self):
        """Автоматически запускает измерение после настройки всех параметров."""
        # Вызываем старт измерения после короткой задержки, чтобы интерфейс успел обновиться
        QTimer.singleShot(500, self.start_distance_calculation) 