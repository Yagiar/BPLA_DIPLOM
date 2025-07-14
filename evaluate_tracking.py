#!/usr/bin/env python3
"""
Оценка эффективности трекинга и сохранения идентификаторов объектов
"""

import cv2
import numpy as np
import supervision as sv
import argparse
import json
import time
import os
from ultralytics import YOLO
from collections import defaultdict, Counter

def load_config():
    """Загрузка конфигурации из settings.json"""
    try:
        with open('settings.json', 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
        return None

def evaluate_tracking(video_path, model_path, conf_threshold=0.25, iou_threshold=0.45, max_frames=5000):
    """
    Оценка эффективности трекинга объектов
    
    Args:
        video_path: путь к видеофайлу
        model_path: путь к модели YOLO
        conf_threshold: порог уверенности для детекций
        iou_threshold: порог IoU для детекций
        max_frames: максимальное количество кадров для обработки
    
    Returns:
        dict: словарь с метриками
    """
    # Загрузка видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return None

    # Загрузка модели
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None

    # Инициализация трекера
    tracker = sv.ByteTrack()
    
    # Структуры данных для анализа
    id_history = defaultdict(list)  # {object_id: [frame_appearances]}
    id_class_consistency = defaultdict(list)  # {object_id: [class_ids]}
    frame_counts = defaultdict(int)  # {object_id: frame_count}
    id_bbox_history = defaultdict(list)  # {object_id: [bbox_positions]}
    confidence_history = defaultdict(list)  # {object_id: [confidence_values]}
    class_counts = Counter()  # Количество объектов каждого класса
    
    # Статистика для визуализации
    track_lengths = []
    lost_tracks = 0
    id_switches = 0
    total_tracks = 0
    previous_detections = None
    
    frame_count = 0
    processing_time = 0
    
    print(f"Начинаем анализ видео {video_path}...")
    
    while frame_count < max_frames:
        start_time = time.time()
        
        # Чтение кадра
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детекция объектов
        results = model.predict(
            frame, 
            conf=conf_threshold, 
            iou=iou_threshold, 
            verbose=False
        )[0]
        
        # Конвертация результатов для трекера
        detections = sv.Detections.from_ultralytics(results)
        
        # Применение трекинга
        detections = tracker.update_with_detections(detections)
        
        # Анализ результатов трекинга
        if detections.tracker_id is not None:
            for i, track_id in enumerate(detections.tracker_id):
                if track_id is None:
                    continue
                    
                # Сохраняем информацию о появлении ID на кадрах
                id_history[track_id].append(frame_count)
                
                # Сохраняем класс объекта
                class_id = int(detections.class_id[i])
                id_class_consistency[track_id].append(class_id)
                
                # Подсчет кадров для каждого ID
                frame_counts[track_id] += 1
                
                # Сохраняем позицию bbox
                id_bbox_history[track_id].append(detections.xyxy[i])
                
                # Подсчет классов
                class_name = results.names[class_id]
                class_counts[class_name] += 1
                
                # Сохраняем значение уверенности
                if detections.confidence is not None:
                    confidence_history[track_id].append(detections.confidence[i])
            
            # Проверка на переключение ID (ID switch)
            if previous_detections is not None and detections.tracker_id is not None:
                # Проверяем IoU между предыдущими и текущими боксами
                for prev_idx, prev_bbox in enumerate(previous_detections.xyxy):
                    prev_id = previous_detections.tracker_id[prev_idx]
                    if prev_id is None:
                        continue
                        
                    # Проверяем, остался ли тот же ID для похожей позиции
                    max_iou = 0
                    max_iou_idx = -1
                    
                    for curr_idx, curr_bbox in enumerate(detections.xyxy):
                        curr_id = detections.tracker_id[curr_idx]
                        if curr_id is None:
                            continue
                            
                        # Вычисляем IoU
                        iou = calculate_iou(prev_bbox, curr_bbox)
                        if iou > max_iou:
                            max_iou = iou
                            max_iou_idx = curr_idx
                    
                    # Если нашли соответствие с высоким IoU, но ID разные
                    if max_iou > 0.5 and max_iou_idx != -1:
                        curr_id = detections.tracker_id[max_iou_idx]
                        if prev_id != curr_id:
                            id_switches += 1
            
            previous_detections = detections
        
        # Измерение времени обработки
        end_time = time.time()
        processing_time += (end_time - start_time)
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Обработано {frame_count} кадров")
    
    cap.release()
    
    # Анализ длительности треков
    for track_id, frames in id_history.items():
        # Проверяем непрерывность трека
        track_length = len(frames)
        consecutive_frames = 1
        max_consecutive = 1
        
        for i in range(1, track_length):
            if frames[i] == frames[i-1] + 1:
                consecutive_frames += 1
            else:
                max_consecutive = max(max_consecutive, consecutive_frames)
                consecutive_frames = 1
        
        max_consecutive = max(max_consecutive, consecutive_frames)
        track_lengths.append(max_consecutive)
        total_tracks += 1
        
        # Проверка консистентности классов
        if len(set(id_class_consistency[track_id])) > 1:
            print(f"ВНИМАНИЕ: Трек ID {track_id} имеет разные классы: {[results.names[c] for c in id_class_consistency[track_id]]}")
    
    # Вычисление потерянных треков
    for track_id, frames in id_history.items():
        if len(frames) < frame_count * 0.1:  # Трек считается потерянным, если он присутствует менее чем в 10% кадров
            lost_tracks += 1
    
    # Расчет средней уверенности по трекам
    confidence_averages = {}
    for track_id, confidences in confidence_history.items():
        if confidences:
            confidence_averages[track_id] = np.mean(confidences)
        else:
            confidence_averages[track_id] = 0.0

    # Расчет метрик
    avg_track_length = np.mean(track_lengths) if track_lengths else 0
    avg_processing_time = processing_time / frame_count if frame_count > 0 else 0
    
    # Формирование результатов
    metrics = {
        "total_frames": frame_count,
        "total_tracks": total_tracks,
        "lost_tracks": lost_tracks,
        "id_switches": id_switches,
        "avg_track_length": avg_track_length,
        "fps": 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
        "class_distribution": dict(class_counts),
        "confidence_averages": confidence_averages,
        "longest_tracks": sorted(frame_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    }
    
    return metrics

def calculate_iou(box1, box2):
    """Вычисление IoU между двумя ограничивающими рамками"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Определение координат пересечения
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    # Проверка на пересечение
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Площадь пересечения
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Площади обоих боксов
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

def visualize_results(video_path, model_path, output_path, metrics, conf_threshold=0.25, iou_threshold=0.45):
    """
    Создание визуализации с метриками трекинга
    
    Args:
        video_path: путь к видеофайлу
        model_path: путь к модели YOLO
        output_path: путь для сохранения выходного видео
        metrics: словарь с метриками
        conf_threshold: порог уверенности для детекций
        iou_threshold: порог IoU для детекций
    """
    # Загрузка видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return
    
    # Параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Настройка записи видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Загрузка модели
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return
    
    # Инициализация трекера и аннотаторов
    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator(
        trace_length=20,  # Длина траектории
        position=sv.Position.BOTTOM_CENTER
    )
    
    # Структуры для хранения информации о треках
    track_info = {}  # {track_id: {'start_frame': frame, 'frames_tracked': count}}
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детекция объектов
        results = model.predict(
            frame, 
            conf=conf_threshold, 
            iou=iou_threshold, 
            verbose=False
        )[0]
        
        # Конвертация результатов для трекера
        detections = sv.Detections.from_ultralytics(results)
        
        # Применение трекинга
        detections = tracker.update_with_detections(detections)
        
        # Создание меток с дополнительной информацией
        labels = []
        
        if detections.tracker_id is not None:
            for i, track_id in enumerate(detections.tracker_id):
                if track_id is None:
                    labels.append("")
                    continue
                
                # Обновляем или создаем информацию о треке
                if track_id not in track_info:
                    track_info[track_id] = {
                        'start_frame': frame_count,
                        'frames_tracked': 0
                    }
                
                track_info[track_id]['frames_tracked'] += 1
                
                # Формируем метку
                class_id = detections.class_id[i]
                class_name = results.names[class_id]
                duration = track_info[track_id]['frames_tracked']
                
                label = f"#{track_id} {class_name} ({duration} frames)"
                labels.append(label)
            
            # Аннотируем кадр
            annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
            annotated_frame = trace_annotator.annotate(annotated_frame, detections=detections)
            
            # Добавляем общую информацию о метриках
            cv2.putText(
                annotated_frame,
                f"Tracks: {metrics['total_tracks']} | ID Switches: {metrics['id_switches']} | Avg Length: {metrics['avg_track_length']:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Записываем кадр
            out.write(annotated_frame)
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Визуализировано {frame_count} кадров")
    
    # Освобождаем ресурсы
    cap.release()
    out.release()
    print(f"Визуализация сохранена в {output_path}")

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Оценка эффективности трекинга объектов')
    parser.add_argument('--video', type=str, help='Путь к видеофайлу')
    parser.add_argument('--model', type=str, help='Путь к модели YOLO')
    parser.add_argument('--conf', type=float, default=0.25, help='Порог уверенности')
    parser.add_argument('--iou', type=float, default=0.45, help='Порог IoU')
    parser.add_argument('--frames', type=int, default=50000, help='Максимальное количество кадров для обработки')
    parser.add_argument('--visualize', action='store_true', help='Создать визуализацию с метриками')
    parser.add_argument('--output', type=str, default='tracking_visualization.mp4', help='Путь для сохранения визуализации')
    
    args = parser.parse_args()
    
    # Если не указаны видео или модель, пытаемся загрузить из конфигурации
    if not args.video or not args.model:
        config = load_config()
        if config:
            if not args.video and 'last_camera' in config:
                args.video = config['last_camera']
                print(f"Используем видео из конфигурации: {args.video}")
                
            if not args.model and 'last_model' in config:
                args.model = config['last_model']
                print(f"Используем модель из конфигурации: {args.model}")
    
    # Проверка наличия необходимых файлов
    if not args.video or not os.path.exists(args.video):
        print("Ошибка: Видеофайл не найден или не указан")
        return
        
    if not args.model or not os.path.exists(args.model):
        print("Ошибка: Файл модели не найден или не указан")
        return
    
    # Оценка эффективности трекинга
    metrics = evaluate_tracking(
        args.video,
        args.model,
        args.conf,
        args.iou,
        args.frames
    )
    
    if metrics:
        # Вывод результатов
        print("\n=== Результаты оценки эффективности трекинга ===")
        print(f"Обработано кадров: {metrics['total_frames']}")
        print(f"Общее количество треков: {metrics['total_tracks']}")
        print(f"Потеряно треков: {metrics['lost_tracks']} ({metrics['lost_tracks']/metrics['total_tracks']*100:.1f}%)")
        print(f"Переключений ID: {metrics['id_switches']}")
        print(f"Средняя длительность трека: {metrics['avg_track_length']:.1f} кадров")
        print(f"Скорость обработки: {metrics['fps']:.1f} FPS")
        
        print("\nРаспределение по классам:")
        for cls, count in metrics['class_distribution'].items():
            print(f"  {cls}: {count}")
        
        print("\nСредняя уверенность по трекам:")
        for track_id, confidence in sorted(metrics['confidence_averages'].items(), key=lambda x: x[1], reverse=True):
            track_length = 0
            for key, value in metrics['longest_tracks']:
                if key == track_id:
                    track_length = value
                    break
            print(f"  ID {track_id}: {confidence:.2f} (длина трека: {track_length} кадров)")
        
        print("\nТоп-10 самых длинных треков:")
        for i, (track_id, length) in enumerate(metrics['longest_tracks']):
            confidence = metrics['confidence_averages'].get(track_id, 0.0)
            print(f"  {i+1}. ID {track_id}: {length} кадров (ср. уверенность: {confidence:.2f})")
        
        # Создание визуализации если указан флаг --visualize
        if args.visualize:
            print("\nСоздание визуализации...")
            visualize_results(
                args.video,
                args.model,
                args.output,
                metrics,
                args.conf,
                args.iou
            )
    
if __name__ == "__main__":
    main() 