#!/usr/bin/env python3
"""
Система мониторинга БПЛА - главный запускающий файл
"""

import sys
from PySide6.QtWidgets import QApplication
from src.widget import Widget

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec()) 