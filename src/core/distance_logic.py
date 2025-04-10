import os
import json
from PySide6.QtWidgets import QMessageBox

class DistanceLogic:
    @staticmethod
    def load_calibration_data():
        calibration_data = {}
        if os.path.exists("calibration_data.json"):
            try:
                with open("calibration_data.json", "r") as f:
                    calibration_data = json.load(f)
            except Exception as e:
                print(f"Ошибка при загрузке данных калибровки: {e}")
        return calibration_data

    @staticmethod
    def load_sync_data():
        sync_data = {}
        if os.path.exists("sync_data.json"):
            try:
                with open("sync_data.json", "r") as f:
                    sync_data = json.load(f)
            except Exception as e:
                print(f"Ошибка при загрузке данных синхронизации: {e}")
        return sync_data

    @staticmethod
    def check_camera_selection(camera1_url, camera2_url):
        if not camera1_url or not camera2_url:
            QMessageBox.warning(None, "Ошибка", "Выберите две разные камеры для измерения.")
            return False
        if camera1_url == camera2_url:
            QMessageBox.warning(None, "Ошибка", "Выберите две разные камеры для измерения.")
            return False
        return True

    @staticmethod
    def check_model_selection(model_path):
        if not model_path:
            QMessageBox.warning(None, "Ошибка", "Необходимо выбрать модель YOLO. Нажмите кнопку 'Выбор модели YOLO'.")
            return False
        return True

    @staticmethod
    def check_warnings(calibration_status, sync_status):
        warnings = []
        if not calibration_status.get('calibrated', False):
            warnings.append("Камеры не откалиброваны")
        if not sync_status.get('synced', False):
            warnings.append("Камеры не синхронизированы")
        if warnings:
            warning_text = "Обнаружены проблемы:\n" + "\n".join([f"- {w}" for w in warnings])
            warning_text += "\n\nИзмерение расстояния может быть неточным. Продолжить?"
            result = QMessageBox.warning(
                None, 
                "Предупреждение", 
                warning_text,
                QMessageBox.Yes | QMessageBox.No
            )
            if result == QMessageBox.No:
                return False
        return True 