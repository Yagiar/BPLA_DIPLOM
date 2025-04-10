import os

class CameraLoader:
    """Class for loading and processing camera lists."""
    
    @staticmethod
    def load_from_file(file_path="cameras.txt"):
        """Load camera list from file.
        
        Args:
            file_path: Path to the camera list file
            
        Returns:
            Tuple containing (camera_urls, camera_names) or (None, None) if file not found
        """
        if not os.path.exists(file_path):
            return None, None
            
        try:
            with open(file_path, "r") as f:
                camera_lines = [line.strip() for line in f if line.strip()]
                
            if not camera_lines:
                return [], []
                
            # Process lines from cameras.txt
            cameras = []
            camera_names = []
            
            for line in camera_lines:
                parts = line.strip().split(' ', 1)  # Split into name and URL
                if len(parts) == 2:
                    camera_name = parts[0]
                    camera_url = parts[1]
                    cameras.append(camera_url)
                    camera_names.append(camera_name)
                else:
                    # If format is incorrect, use the entire string as URL
                    cameras.append(line)
                    camera_names.append(f"Камера {len(cameras)}")
                    
            return cameras, camera_names
            
        except Exception as e:
            print(f"Error loading cameras: {e}")
            return [], []
    
    @staticmethod
    def find_camera_index_by_url(camera_combos, camera_url):
        """Find the index of a camera in a QComboBox by its URL.
        
        Args:
            camera_combos: QComboBox containing cameras
            camera_url: URL to search for
            
        Returns:
            Index of the camera or -1 if not found
        """
        for i in range(camera_combos.count()):
            if camera_combos.itemData(i) == camera_url:
                return i
        return -1
    
    @staticmethod
    def populate_comboboxes(cameras, camera_names, *comboboxes):
        """Populate multiple comboboxes with camera data.
        
        Args:
            cameras: List of camera URLs
            camera_names: List of camera names
            *comboboxes: QComboBox widgets to populate
        """
        for combo in comboboxes:
            combo.clear()
            for url, name in zip(cameras, camera_names):
                combo.addItem(name, url) 