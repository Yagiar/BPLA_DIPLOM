
# Инициализация детектора YOLO
def init_detector(model_path, conf_threshold, iou_threshold):
    model = load_model_YOLOv8(model_path)
    model.conf = conf_threshold
    model.iou = iou_threshold
    return model

# Детектирование объектов на кадре
def detect_objects(model, frame):
    results = model.predict(frame, verbose=False)[0]
    
    detections = {
        "boxes": [box.xyxy[0] for box in results.boxes],
        "scores": [box.conf[0] for box in results.boxes],
        "class_ids": [int(box.cls[0]) for box in results.boxes],
        "class_names": [results.names[int(box.cls[0])] for box in results.boxes]
    }
    
    return detections

# Аннотирование кадра с результатами детекции
def annotate_frame(frame, detections):
    annotated_frame = frame.copy()
    
    for i in range(len(detections["boxes"])):
        box = detections["boxes"][i]
        score = detections["scores"][i]
        class_name = detections["class_names"][i]
        
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = get_color_for_class(detections["class_ids"][i])
        
        draw_rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        draw_text(annotated_frame, f"{class_name}: {score:.2f}", (x1, y1-10), color)
    
    return annotated_frame

# Основной цикл обработки видео
def process_video(video_source, model_path, conf=0.25, iou=0.45):
    detector = init_detector(model_path, conf, iou)
    video = open_video_stream(video_source)
    
    while video.is_open():
        success, frame = video.read_frame()
        if not success:
            break
        
        detections = detect_objects(detector, frame)
        annotated_frame = annotate_frame(frame, detections)
        display_frame(annotated_frame)
        
        if key_pressed('q'):
            break
    
    video.release()