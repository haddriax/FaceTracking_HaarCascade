from typing import List, Dict, Any, Tuple

import cv2
import json
import logging

import numpy as np


class FaceDetector:
    """
    A class for detecting faces in a video using Haar cascades.
    Uses a config json file.

    Attributes:
        cascades_config (List): A list of configurations for each Haar cascades used.
        general_config (Dict): General configuration about the video and for the processing.
        source_video (cv2.VideoCapture): The source video to process.
        frame_rate (int): The frame rate of the source video.
        frame_total (int): The total number of frames in the source video.
        frame_width (int): The width of each video frame in pixels.
        frame_height (int): The height of each video frame in pixels.
        detections_by_cascades (Dict[str, List[Dict[str, Any]]]): All the [x, y, w, h] detected per cascade for each frame.
    """
    def __init__(self, config_file: str, detection_file: str | None = None ):
        """
        Initializes the FaceDetector class.

        Args:
            config_file (str): Path to the JSON configuration file.
            detection_file (str | None, optional): Path to an optional detection file.
        """

        self.cascades_config: List = []
        self.general_config: Dict = {}
        self.source_video: cv2.VideoCapture | None = None
        self.frame_rate: int = 0
        self.frame_total: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.detections_by_cascades: Dict[str, List[Dict[str, Any]]] | None = None
        self.cascade: List[cv2.CascadeClassifier] = []
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

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
        assert self.frame_total > 0
        self.logger.info(f"Video info: {self.frame_total} frames "
                         f"{self.frame_width}x{self.frame_height}@{self.frame_rate}fps")

    def _prepare_cascades(self) -> dict[str: cv2.CascadeClassifier]:
        """
        Generate all the CascadeClassifier objects if they are activated in the config file
        Returns:
            dict[str: cv2.CascadeClassifier]: A dict with the name of the cascade as key and classifier as value.
        """
        self.cascades = {
            config["name"]: cv2.CascadeClassifier(self.general_config["cascades_dir"]+config["cascade_path"])
                for config in self.cascades_config if
                config["enabled"]
        }
        self.logger.info(f"Loaded cascades: {[k for k in self.cascades.keys()]}")
        return self.cascades

    def _detect_on_video(self):
        """
        Run the full detection on every frame with all enabled Haar cascades on the video.
        """
        self.logger.info(f"Starting Haar cascade detection...")
        for i in range(self.frame_total):
            ret, frame = self.source_video.read()
            self.logger.debug(f"Processing frame {i + 1}/{self.frame_total}.")
            if frame is None or not ret:
                break

            # Process the frame into grayscale to run Haar cascade.
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.mean(frame_gray) < 50:
                frame_gray = cv2.equalizeHist(frame_gray)

    def _detect_on_frame(self, frame_gray: np.ndarray):
        for config in self.cascades:
            if config["enabled"]:
                detections, reject_levels, level_weights = self._detect_faces(
                    self.cascade[config["name"]],
                    frame_gray,
                    config["scale_factor"],
                    config["min_neighbors"],
                    tuple(config["min_size"])
                )
                self.detections_by_cascades[config["name"]].append({
                    "detections": detections.tolist() if len(detections) > 0 else [],
                    "reject_levels": list(reject_levels) if reject_levels is not None else [],
                    "weights": list(level_weights) if level_weights is not None else []
                })

    def _detect_faces(self,
                      cascade: cv2.CascadeClassifier,
                      frame_gray: np.ndarray,
                      scale_factor: float,
                      min_neighbors: int,
                      min_size: Tuple[int, int]):
        try:
            faces, reject_levels, level_weights = cascade.detectMultiScale3(
                frame_gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                outputRejectLevels=True
            )
        except cv2.error as e:
            self.logger.error(f"Error with detectMultiScale3: {e}")
            # Return a default multiscale.
            faces, reject_levels, level_weights = cascade.detectMultiScale(
                frame_gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            ), None, None
        return faces, reject_levels, level_weights

    def _load_detections(self):
        """
        Load the matching detection file if it exists, and set it into self.detections_by_cascades.
        Returns:
            bool: True if the loading succeeded, False otherwise.
        """
        # @todo: Loading from file
        if self.detections_by_cascades is None:
            return False

    def run(self):
        if not self._load_detections():
            self._prepare_cascades()
            self._detect_on_video()
            self.source_video.set(cv2.CAP_PROP_POS_FRAMES, 0)


if __name__ == '__main__':
    f = FaceDetector(config_file="config.json")
    f.run()