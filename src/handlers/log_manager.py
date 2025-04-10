from PySide6.QtWidgets import QTextEdit

class LogManager:
    def __init__(self, detection_log: QTextEdit, distance_log: QTextEdit):
        self.detection_log = detection_log
        self.distance_log = distance_log

    def log_message(self, message, color="black", both_logs=False):
        """Adds a message to the log with color formatting.
        
        Args:
            message: The text of the message
            color: The color of the text
            both_logs: If True, displays the message in both logs, regardless of the current mode
        """
        formatted_message = f'<span style="color: {color};">{message}</span>'
        
        if both_logs or self.detection_log:
            self.detection_log.append(formatted_message)
            
        if both_logs or self.distance_log:
            self.distance_log.append(formatted_message) 