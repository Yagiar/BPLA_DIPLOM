import json
import os

class Config:
    def __init__(self):
        self.config_file = "settings.json"
        self.default_settings = {
            'model': {
                'path': None,
                'conf': 0.25,
                'iou': 0.45,
                'device': 'cpu',
                'half': False
            },
            'tracker': {
                'fps': 30
            }
        }
        self.settings = self.load_settings()
    
    def load_settings(self):
        """Загружает настройки из файла."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return self.default_settings.copy()
        return self.default_settings.copy()
    
    def save_settings(self):
        """Сохраняет настройки в файл."""
        with open(self.config_file, 'w') as f:
            json.dump(self.settings, f, indent=4)
    
    def update_settings(self, new_settings):
        """Обновляет настройки."""
        if 'conf' in new_settings:
            self.settings['model']['conf'] = new_settings['conf']
        if 'iou' in new_settings:
            self.settings['model']['iou'] = new_settings['iou']
        if 'device' in new_settings:
            self.settings['model']['device'] = new_settings['device']
        if 'half' in new_settings:
            self.settings['model']['half'] = new_settings['half']
        if 'fps' in new_settings:
            self.settings['tracker']['fps'] = new_settings['fps']
        self.save_settings()
    
    def update_model_path(self, path):
        """Обновляет путь к модели."""
        self.settings['model']['path'] = path
        self.save_settings()
    
    def get_model_settings(self):
        """Возвращает настройки модели."""
        return self.settings['model']
    
    def get_tracker_settings(self):
        """Возвращает настройки трекера."""
        return self.settings['tracker'] 