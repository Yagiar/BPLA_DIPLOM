import cv2
import numpy as np
import time
from datetime import datetime

# URL камер (измените на свои)
cam1_url = "http://192.168.0.101:4747/video"
cam2_url = "http://192.168.0.109:4747/video"

# Открываем видеопотоки
cap1 = cv2.VideoCapture(cam1_url)
cap2 = cv2.VideoCapture(cam2_url)

# Устанавливаем ограничение FPS до 29
cap1.set(cv2.CAP_PROP_FPS, 29)
cap2.set(cv2.CAP_PROP_FPS, 29)

if not cap1.isOpened() or not cap2.isOpened():
    print("Ошибка: Не удалось открыть одну или обе камеры")
    exit()

# Задаем порог яркости для определения вспышки
brightness_threshold = 100

# Счетчики кадров для каждой камеры
frame_count_cam1 = 0
frame_count_cam2 = 0

# Флаги обнаружения вспышек
flash_detected_cam1 = False
flash_detected_cam2 = False

# Счетчики вспышек
flash_count_cam1 = 0
flash_count_cam2 = 0

# Флаг для указания, что счетчики уже были сброшены после вспышки
counters_reset_after_flash = False

# Время старта отсчета после сброса счетчиков
time_since_reset = None

# Вывод статистики через определенные интервалы
last_report_30sec = 0
last_report_60sec = 0
last_report_120sec = 0

# Списки для хранения истории расхождения кадров
frame_diff_history = []

# Переменная для контроля частоты кадров
prev_time = time.time()
frame_interval = 1 / 29.0  # Интервал для 29 FPS

print("Начинаем захват видео с двух камер. Нажмите 'q' для выхода.")

while True:
    # Контроль частоты кадров
    current_time = time.time()
    time_elapsed = current_time - prev_time
    
    # Захват кадров с обеих камер
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Ошибка чтения кадров с одной или обеих камер")
        break
    
    # Преобразуем кадры в оттенки серого
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Вычисляем среднюю яркость кадров
    avg_brightness1 = np.mean(gray1)
    avg_brightness2 = np.mean(gray2)
    
    # Получаем текущую временную метку
    current_time = datetime.now()
    timestamp = current_time.strftime("%H:%M:%S.%f")[:-3]
    
    # Проверяем вспышку для камеры 1
    if avg_brightness1 > brightness_threshold:
        if not flash_detected_cam1:  # Чтобы не считать одну вспышку несколько раз
            flash_detected_cam1 = True
            flash_count_cam1 += 1
            print(f"[{timestamp}] Камера 1: Обнаружена вспышка #{flash_count_cam1}, frame={frame_count_cam1}")
    else:
        flash_detected_cam1 = False
        frame_count_cam1 += 1
    
    # Проверяем вспышку для камеры 2
    if avg_brightness2 > brightness_threshold:
        if not flash_detected_cam2:  # Чтобы не считать одну вспышку несколько раз
            flash_detected_cam2 = True
            flash_count_cam2 += 1
            print(f"[{timestamp}] Камера 2: Обнаружена вспышка #{flash_count_cam2}, frame={frame_count_cam2}")
    else:
        flash_detected_cam2 = False
        frame_count_cam2 += 1
    
    # Проверяем, нужно ли сбросить счетчики
    if flash_count_cam1 > 0 and flash_count_cam2 > 0 and not counters_reset_after_flash:
        print(f"\n[{timestamp}] Обнаружены вспышки на обеих камерах, сбрасываем счетчики кадров")
        frame_count_cam1 = 0
        frame_count_cam2 = 0
        counters_reset_after_flash = True
        time_since_reset = time.time()
        print(f"Начинаем отсчет времени для мониторинга расхождения кадров\n")
    
    # Если счетчики были сброшены, следим за расхождением кадров
    if counters_reset_after_flash and time_since_reset is not None:
        elapsed_seconds = time.time() - time_since_reset
        
        # Отчет каждые 30 секунд
        if elapsed_seconds >= 30 * (last_report_30sec + 1):
            last_report_30sec += 1
            frame_diff = frame_count_cam1 - frame_count_cam2
            print(f"\n===== ОТЧЕТ ПО РАСХОЖДЕНИЮ КАДРОВ (через {30 * last_report_30sec} сек) =====")
            print(f"Камера 1: {frame_count_cam1} кадров")
            print(f"Камера 2: {frame_count_cam2} кадров")
            print(f"Разница: {frame_diff} кадров")
            print(f"Скорость расхождения: {frame_diff / elapsed_seconds:.2f} кадров/сек")
            print("=====================================================\n")
            
            # Сохраняем данные для анализа
            frame_diff_history.append({
                "time": elapsed_seconds,
                "cam1_frames": frame_count_cam1,
                "cam2_frames": frame_count_cam2,
                "diff": frame_diff
            })
        
        # Отчет каждые 60 секунд
        if elapsed_seconds >= 60 * (last_report_60sec + 1):
            last_report_60sec += 1
            frame_diff = frame_count_cam1 - frame_count_cam2
            print(f"\n===== ОТЧЕТ ПО РАСХОЖДЕНИЮ КАДРОВ (через {60 * last_report_60sec} сек) =====")
            print(f"Камера 1: {frame_count_cam1} кадров")
            print(f"Камера 2: {frame_count_cam2} кадров")
            print(f"Разница: {frame_diff} кадров")
            print(f"Скорость расхождения: {frame_diff / elapsed_seconds:.2f} кадров/сек")
            print("=====================================================\n")
        
        # Отчет каждые 120 секунд
        if elapsed_seconds >= 120 * (last_report_120sec + 1):
            last_report_120sec += 1
            frame_diff = frame_count_cam1 - frame_count_cam2
            print(f"\n===== ОТЧЕТ ПО РАСХОЖДЕНИЮ КАДРОВ (через {120 * last_report_120sec} сек) =====")
            print(f"Камера 1: {frame_count_cam1} кадров")
            print(f"Камера 2: {frame_count_cam2} кадров")
            print(f"Разница: {frame_diff} кадров")
            print(f"Скорость расхождения: {frame_diff / elapsed_seconds:.2f} кадров/сек")
            print("=====================================================\n")
    
    # Отображаем кадры (опционально)
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)
    
    # Ожидание для достижения 29 FPS
    if time_elapsed < frame_interval:
        time.sleep(frame_interval - time_elapsed)
    
    prev_time = time.time()
    
    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Выводим итоговую статистику при завершении
print("\n===== ИТОГОВАЯ СТАТИСТИКА =====")
print(f"Всего вспышек на камере 1: {flash_count_cam1}")
print(f"Всего вспышек на камере 2: {flash_count_cam2}")

if counters_reset_after_flash:
    total_elapsed = time.time() - time_since_reset
    frame_diff_final = frame_count_cam1 - frame_count_cam2
    
    print(f"\nИтоговое время наблюдения: {total_elapsed:.1f} секунд")
    print(f"Камера 1: захвачено {frame_count_cam1} кадров ({frame_count_cam1/total_elapsed:.1f} к/с)")
    print(f"Камера 2: захвачено {frame_count_cam2} кадров ({frame_count_cam2/total_elapsed:.1f} к/с)")
    print(f"Итоговая разница: {frame_diff_final} кадров")
    print(f"Средняя скорость расхождения: {frame_diff_final/total_elapsed:.2f} кадров/сек")
    
    # Если есть достаточно данных, анализируем изменение скорости расхождения
    if len(frame_diff_history) > 1:
        first_diff = frame_diff_history[0]["diff"]
        first_time = frame_diff_history[0]["time"]
        last_diff = frame_diff_history[-1]["diff"]
        last_time = frame_diff_history[-1]["time"]
        
        drift_rate_change = (last_diff/last_time) - (first_diff/first_time)
        print(f"\nИзменение скорости расхождения: {drift_rate_change:.4f} кадров/сек²")
        
        if abs(drift_rate_change) < 0.01:
            print("✅ Стабильное расхождение: скорость расхождения постоянна")
        else:
            print("⚠️ Нестабильное расхождение: скорость расхождения меняется со временем")

# Освобождаем ресурсы
cap1.release()
cap2.release()
cv2.destroyAllWindows()
