import json
import os
import time
from pathlib import Path

class Config:
    """
    Класс для работы с конфигурацией приложения.
    Хранит настройки в файле settings.json.
    """
    def __init__(self, config_file="settings.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self):
        """Загружает конфигурацию из файла."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Ошибка при чтении файла конфигурации: {e}")
                return self._create_default_config()
        else:
            return self._create_default_config()
    
    def _create_default_config(self):
        """Создает конфигурацию по умолчанию."""
        default_config = {
            "model": {
                "path": "",
                "conf": 0.25,
                "iou": 0.45,
                "device": "cpu",
                "half": False
            },
            "last_camera": "",
            "last_model": "",
            "ui": {
                "theme": "light",
                "font_size": 10,
                "window_size": [1200, 800]
            },
            "cameras": {
                "calibrated": False,
                "calibrated_pairs": []
            },
            "sync": {
                "is_synced": False,
                "synced_cameras": []
            },
            "detection": {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "show_labels": True,
                "show_boxes": True
            },
            "distance_measure": {
                "enabled": False,
                "baseline": 10.0,
                "cameras": []
            }
        }
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config=None):
        """Сохраняет конфигурацию в файл."""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка при сохранении файла конфигурации: {e}")
    
    def update_config(self):
        """Обновляет файл конфигурации."""
        self._save_config()
    
    def get_last_camera(self):
        """Возвращает URL последней использованной камеры."""
        return self.config.get("last_camera", "")
    
    def set_last_camera(self, camera_url):
        """Устанавливает URL последней использованной камеры."""
        self.config["last_camera"] = camera_url
        self.update_config()
    
    def get_last_model(self):
        """Возвращает путь к последней использованной модели."""
        return self.config.get("last_model", "")
    
    def set_last_model(self, model_path):
        """Устанавливает путь к последней использованной модели."""
        self.config["last_model"] = model_path
        self.update_config()
    
    def get_detection_settings(self):
        """Возвращает настройки детекции."""
        return self.config.get("detection", {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "show_labels": True,
            "show_boxes": True
        })
    
    def set_detection_settings(self, confidence_threshold=None, iou_threshold=None, 
                               show_labels=None, show_boxes=None):
        """Устанавливает настройки детекции."""
        if "detection" not in self.config:
            self.config["detection"] = {}
        
        if confidence_threshold is not None:
            self.config["detection"]["confidence_threshold"] = confidence_threshold
        
        if iou_threshold is not None:
            self.config["detection"]["iou_threshold"] = iou_threshold
        
        if show_labels is not None:
            self.config["detection"]["show_labels"] = show_labels
        
        if show_boxes is not None:
            self.config["detection"]["show_boxes"] = show_boxes
        
        self.update_config()
    
    def get_ui_settings(self):
        """Возвращает настройки интерфейса."""
        return self.config.get("ui", {
            "theme": "light",
            "font_size": 10,
            "window_size": [1200, 800]
        })
    
    def set_ui_settings(self, theme=None, font_size=None, window_size=None):
        """Устанавливает настройки интерфейса."""
        if "ui" not in self.config:
            self.config["ui"] = {}
        
        if theme is not None:
            self.config["ui"]["theme"] = theme
        
        if font_size is not None:
            self.config["ui"]["font_size"] = font_size
        
        if window_size is not None:
            self.config["ui"]["window_size"] = window_size
        
        self.update_config()
    
    def is_cameras_calibrated(self):
        """Возвращает True, если камеры калиброваны."""
        return self.config.get("cameras", {}).get("calibrated", False)
    
    def get_calibrated_camera_pairs(self):
        """Возвращает список калиброванных пар камер."""
        return self.config.get("cameras", {}).get("calibrated_pairs", [])
    
    def update_calibration_status(self, is_calibrated, camera_pair=None):
        """Обновляет статус калибровки камер."""
        if "cameras" not in self.config:
            self.config["cameras"] = {"calibrated": False, "calibrated_pairs": []}
        
        self.config["cameras"]["calibrated"] = is_calibrated
        
        if camera_pair and is_calibrated:
            # Добавляем пару камер в список, если её там ещё нет
            pairs = self.config["cameras"]["calibrated_pairs"]
            if camera_pair not in pairs:
                pairs.append(camera_pair)
                self.config["cameras"]["calibrated_pairs"] = pairs
        
        self.update_config()
    
    def is_cameras_synced(self):
        """Возвращает True, если камеры синхронизированы."""
        return self.config.get("sync", {}).get("is_synced", False)
    
    def get_synced_cameras(self):
        """Возвращает список синхронизированных камер."""
        return self.config.get("sync", {}).get("synced_cameras", [])
    
    def update_sync_status(self, is_synced, synced_cameras=None):
        """Обновляет статус синхронизации камер."""
        if "sync" not in self.config:
            self.config["sync"] = {"is_synced": False, "synced_cameras": []}
        
        self.config["sync"]["is_synced"] = is_synced
        
        if synced_cameras and is_synced:
            self.config["sync"]["synced_cameras"] = synced_cameras
        
        self.update_config()
    
    def get_distance_measure_settings(self):
        """Возвращает настройки измерения расстояния."""
        return self.config.get("distance_measure", {
            "enabled": False,
            "baseline": 10.0,
            "cameras": []
        })
    
    def update_distance_measure_settings(self, enabled, baseline=None, cameras=None):
        """Обновляет настройки измерения расстояния."""
        if "distance_measure" not in self.config:
            self.config["distance_measure"] = {
                "enabled": False,
                "baseline": 10.0,
                "cameras": []
            }
        
        self.config["distance_measure"]["enabled"] = enabled
        
        if baseline is not None:
            self.config["distance_measure"]["baseline"] = baseline
            
        if cameras is not None:
            self.config["distance_measure"]["cameras"] = cameras
            
        self.update_config()
    
    def get_calibration_status(self):
        """Возвращает статус калибровки."""
        return self.config.get("cameras", {"calibrated": False})
    
    def get_sync_status(self):
        """Возвращает статус синхронизации."""
        return self.config.get("sync", {"synced": False})
    
    def get_model_path(self):
        """Возвращает путь к модели из секции model."""
        if "model" not in self.config:
            self.config["model"] = {"path": ""}
            
        return self.config["model"].get("path", "")
        
    def set_model_path(self, model_path):
        """Устанавливает путь к модели в секции model."""
        if "model" not in self.config:
            self.config["model"] = {"path": ""}
            
        self.config["model"]["path"] = model_path
        
        # Для совместимости обновим и last_model
        self.config["last_model"] = model_path
        
        self.update_config()
        
    def get_model_settings(self):
        """Возвращает все настройки модели."""
        return self.config.get("model", {
            "path": "",
            "conf": 0.25,
            "iou": 0.45,
            "device": "cpu",
            "half": False
        })
        
    def set_model_settings(self, path=None, conf=None, iou=None, device=None, half=None):
        """Устанавливает настройки модели."""
        if "model" not in self.config:
            self.config["model"] = {
                "path": "",
                "conf": 0.25,
                "iou": 0.45,
                "device": "cpu",
                "half": False
            }
        
        if path is not None:
            self.config["model"]["path"] = path
            # Для совместимости обновим и last_model
            self.config["last_model"] = path
            
        if conf is not None:
            self.config["model"]["conf"] = conf
            
        if iou is not None:
            self.config["model"]["iou"] = iou
            
        if device is not None:
            self.config["model"]["device"] = device
            
        if half is not None:
            self.config["model"]["half"] = half
            
        self.update_config()


if __name__ == "__main__":
    # Тестирование класса
    config = Config()
    config.set_last_camera("rtsp://example.com:554/stream1")
    config.set_last_model("yolov8n.pt")
    config.set_detection_settings(confidence_threshold=0.3)
    config.update_calibration_status(True, ["rtsp://cam1", "rtsp://cam2"])
    config.update_sync_status(True, ["rtsp://cam1", "rtsp://cam2"])
    config.update_distance_measure_settings(True, 15.0, ["rtsp://cam1", "rtsp://cam2"])
    print("Настройки сохранены.") 