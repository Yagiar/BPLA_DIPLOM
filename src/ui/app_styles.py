class AppStyles:
    """Class for managing application styles."""
    
    @staticmethod
    def get_main_stylesheet():
        """Get the main application stylesheet."""
        return """
        QWidget {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #2C3E50, stop:1 #3498DB);
            font-family: Arial, sans-serif;
        }
        
        /* Title style */
        QLabel#title {
            color: white;
            font-size: 24px;
            font-weight: bold;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            margin: 10px;
        }
        
        QPushButton {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #2ECC71, stop:1 #27AE60);
            color: white;
            border: none;
            padding: 12px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin: 5px;
            transition-duration: 0.4s;
            cursor: pointer;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            min-height: 30px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #27AE60, stop:1 #229954);
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
        }
        
        QPushButton:pressed {
            background-color: #229954;
            transform: translateY(1px);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        QLabel#video_label {
            background-color: black;
            border: 3px solid #2ECC71;
            border-radius: 15px;
            padding: 10px;
            margin: 10px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            min-width: 320px;
            min-height: 240px;
        }
        
        QTextEdit {
            background-color: rgba(255, 255, 255, 0.85);
            border: 2px solid #2ECC71;
            border-radius: 10px;
            padding: 10px;
            margin: 5px;
            font-size: 12px;
            color: #2C3E50;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            min-height: 100px;
        }
        
        QScrollBar:vertical {
            border: none;
            background-color: rgba(0, 0, 0, 0.1);
            width: 10px;
            border-radius: 5px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #2ECC71;
            border-radius: 5px;
            min-height: 30px;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QGroupBox {
            background-color: rgba(0, 0, 0, 0.15);
            border: 1px solid #2ECC71;
            border-radius: 10px;
            padding: 10px;
            margin: 5px;
            color: white;
            font-weight: bold;
            min-height: 50px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: white;
            font-weight: bold;
        }
        
        QRadioButton {
            color: white;
            font-size: 14px;
            spacing: 8px;
            margin: 5px;
            color: #FFFFFF;
            font-size: 14px;
            background: transparent;
            padding: 2px;
        }
        
        QRadioButton::indicator {
            width: 18px;
            height: 18px;
        }
        
        QComboBox {
            background-color: rgba(255, 255, 255, 0.8);
            border: 1px solid #2ECC71;
            border-radius: 5px;
            padding: 5px;
            min-width: 150px;
            color: #2C3E50;
        }
        
        QComboBox:hover {
            border: 1px solid #27AE60;
        }
        
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left-width: 1px;
            border-left-color: #2ECC71;
            border-left-style: solid;
        }
        
        QScrollArea {
            border: none;
            background-color: transparent;
        }

        QLabel {
            color: #FFFFFF;
            font-size: 14px;
            background: transparent;
            padding: 2px;
        }
        """ 