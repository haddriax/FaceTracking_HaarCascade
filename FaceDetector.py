import cv2
import json
import logging

class FaceDetector:
    """
    A class for detecting faces in a video using Haar cascades.
    Uses a config json file.

    Attributes:
        cascades_config (list): A list of configurations for each Haar cascades used.
        general_config (dict): General configuration about the video and for the processing.
        source_video (cv2.VideoCapture): The source video to process.
        frame_rate (int): The frame rate of the source video.
        frame_total (int): The total number of frames in the source video.
        frame_width (int): The width of each video frame in pixels.
        frame_height (int): The height of each video frame in pixels.
    """

    cascades_config: list
    general_config: dict
    source_video: cv2.VideoCapture | None
    frame_rate: int
    frame_total: int
    frame_width: int
    frame_height: int
    logger: logging.Logger

    def __init__(self, config_file: str, detection_file: str | None = None ):
        """
        Initializes the FaceDetector class.

        Args:
            config_file (str): Path to the JSON configuration file.
            detection_file (str | None, optional): Path to an optional detection file.
        """

        self.cascades_config: list = []
        self.general_config: dict = {}
        self.source_video: cv2.VideoCapture = None
        self.frame_rate: int = 0
        self.frame_total: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level=logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        try:
            with open(config_file, "r") as f:
                c = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Configuration file '{config_file}' not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Configuration file '{config_file}' is not a valid JSON file.")

        self.cascades_config = c["cascades_config"]
        self.general_config = c["general_config"]

        self._load_video()

        self._get_video_info()

        self.logger.info(f"Successfully initialized FaceDetector.")

    def _load_video(self):
        self.source_video = cv2.VideoCapture(self.general_config["source_video"])
        if not self.source_video.isOpened():
            raise ValueError(f"Could not open video file: {self.general_config['source_video']}")

    def _get_video_info(self):
        """
        Retrieves and sets video information such as frame rate, total frames, width, and height.
        """
        self.frame_rate   = int(self.source_video.get(cv2.CAP_PROP_FPS))
        self.frame_total  = int(self.source_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width  = int(self.source_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.source_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info(f"Video info: {self.frame_total} frames "
                         f"{self.frame_width}x{self.frame_height}@{self.frame_rate}fps")

    def _detect_faces(self):
        pass

if __name__ == '__main__':
    f = FaceDetector(config_file="config.json")