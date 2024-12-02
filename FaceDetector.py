import os
from typing import List, Dict, Any, Tuple, Sequence

import cv2
import json
import logging

import numpy as np

from FaceTracker import FaceTracker


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
        all_detections_by_cascades (List[Dict[str, Any]]): All the [x, y, w, h] detected per cascade for each frame.
    """

    def __init__(self, config_file: str, detection_file: str | None = None):
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
        self.all_detections_by_cascades: List[Dict[str, Any]] | None = None
        self.cascades: Dict[str: Dict[str: cv2.CascadeClassifier, str, str, str]] = {}
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self.logger.setLevel(level=logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        try:
            with open(config_file, "r") as file:
                json_file = json.load(file)
        except FileNotFoundError as e:
            self.logger.error(f"{e}. Configuration file '{config_file}' not found. Exiting program.")
            exit(1)
        except json.JSONDecodeError as e:
            self.logger.error(f"Configuration file '{config_file}' is not a valid JSON file. {e}. Exiting program.")
            exit(1)

        self.cascades_config = json_file["cascades_config"]
        self.general_config = json_file["general_config"]

        try:
            self._load_video()
        except ValueError as e:
            self.logger.error(f"{e}. Exiting program.")
            exit(1)

        self._get_video_info()

        self.logger.info(f"Successfully initialized FaceDetector.")

    def _load_video(self):
        """
        Load the video from the path in the config file and store it into to the self.source_video.
        Attempt loading using cv2.VideoCapture.
        Raises:
            ValueError: If the video file cannot be opened.
        """
        self.source_video = cv2.VideoCapture(self.general_config["source_video"])
        if not self.source_video.isOpened():
            raise ValueError(f"Could not open video file: {self.general_config['source_video']}")

    def _get_video_info(self):
        """
        Retrieves and sets video information such as frame rate, total frames, width, and height.
        """
        self.frame_rate = int(self.source_video.get(cv2.CAP_PROP_FPS))
        self.frame_total = int(self.source_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.source_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.source_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.frame_total > 0
        self.logger.info(f"Source video: '{self.general_config['source_video']}' {self.frame_total} frames "
                         f"{self.frame_width}x{self.frame_height}@{self.frame_rate}fps")

    def _prepare_cascades(self) -> dict[str: cv2.CascadeClassifier]:
        """
        Generate all the CascadeClassifier objects if they are activated in the config file

        Returns:
            dict[str: cv2.CascadeClassifier] with the name of the cascade as key and classifier as value.
        """
        self.cascades = {
            config["name"]:
                {
                    "classifier": cv2.CascadeClassifier(self.general_config["cascades_dir"] + config["cascade_path"]),
                    "scale_factor": config["scale_factor"],
                    "min_neighbors": config["min_neighbors"],
                    "min_size": config["min_size"]
                }
            for config in self.cascades_config if
            config["enabled"]
        }
        self.logger.info(f"Loaded cascades: {[k for k in self.cascades.keys()]}")
        return self.cascades

    def _process_detections_frame(self, frame_detections) -> List[Dict]:
        """
        Given the detection dictionary containing the detections form one frame.
        - Remove the detections that don't match the threshold specified in the config file, by cascade.
        - Merge the detections that are too close to each other, regardless of the cascade it comes from.
        Args:
            frame_detections (Dict): A Dict with cascade name as keys and a List of detections (Dict)
        Returns:
            A list of filtered detections, now unrelated to cascades.
        """
        # Remove detection if under the desired weight
        filtered_detections = self._filter_by_weight_frame(frame_detections)
        # Merge detection if their coordinates are too close, based on cascade thresholds.
        filtered_detections = self._filter_centers_frame(frame_detections)
        return filtered_detections

    def _process_video_frame(self, source_frame, frame_detections, face_tracker):
        # Step 1: Filter and merge detections
        filtered_detections = self._process_detections_frame(frame_detections)

        # Extract bounding boxes for tracker
        detections = [detection["detections"] for detection in filtered_detections]

        # Step 2: Update tracker with filtered detections
        tracked_faces = face_tracker.update(detections)

        # Step 3: Blur detected regions
        blur_zones = [{"detections": track_data["bbox"]} for track_data in tracked_faces.values()]
        blurred_frame = self._blur_frame(source_frame, blur_zones)

        #detections = []
        #for cascade_name, cascade_results in frame_detections.items():
         #   for detection_entry in cascade_results:
          #      detections.extend(detection_entry["detections"])

        #tracked_faces = face_tracker.update(detections)

        #blur_zones = [{"detections": track_data["bbox"]} for track_data in tracked_faces.values()]
        #blurred_frame = self._blur_frame(source_frame, blur_zones)

        blurred_frame = self._display_track_path(blurred_frame, tracked_faces, face_tracker)
        return blurred_frame

    def _display_track_path(self, frame, tracked_faces, face_tracker: FaceTracker):
        """
        Compute and show the track path on the video.
        """
        for track_id, track_data in tracked_faces.items():
            x, y, w, h = track_data["bbox"]
            center = (x + w // 2, y + h // 2)

            tp = face_tracker.tracking_paths

            # Update or initialize the tracking path
            if track_id not in tp:
                tp[track_id] = []
            tp[track_id].append(center)
            for j in range(1, len(tp[track_id])):
                if tp[track_id][j - 1] and tp[track_id][j]:
                    cv2.line(frame, tp[track_id][j - 1], tp[track_id][j], (0, 255, 0), 2)
        return frame

    def _detect_on_video(self):
        """
        Run detection and tracking on every frame using Haar cascades.
        Blurs specified regions and tracks faces with movement visualization.
        """
        self.logger.info(f"Starting Haar cascade detection, tracking, and movement visualization...")

        # @todo should be in config file.
        face_tracker = FaceTracker(max_before_untrack=75, distance_threshold=70)

        output_path = self.general_config.get("output_video_path", "output_tracked_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, self.frame_rate, (self.frame_width, self.frame_height))

        for i in range(self.frame_total):
            ret, source_frame = self.source_video.read()
            self.logger.debug(f"Processing frame {i + 1}/{self.frame_total}.")
            if source_frame is None or not ret:
                break

            frame_gray = self._preprocess_frame(source_frame)
            frame_detections = self._detect_on_frame(frame_gray)
            processed_frame = self._process_video_frame(source_frame, frame_detections, face_tracker)
            output_video.write(processed_frame)

        output_video.release()
        self.logger.info(f"Finished processing video. Output saved to {output_path}.")

    def _save_results_to_json(self):
        """
        Serializes the detection data and saves it to a JSON file.
        If the file exists, it is overwritten.
        File name and path are set in the config file: output_json_dir and output_json_name

        Raises:
            FileNotFoundError: If the provided file path does not exist.
            PermissionError: If the user does not have permission to write to the file.
            json.JSONDecodeError: If there is an issue with serializing the data into JSON format.
            Exception: For any other unexpected errors that may occur during the file handling or serialization process.
        """
        json_result_full_path: str = self.general_config["output_json_dir"]+self.general_config["output_json_name"]

        try:
            with open(json_result_full_path, "w") as json_file:
                detections_data_serializable = self.convert_to_serializable(self.all_detections_by_cascades)
                json.dump(detections_data_serializable, json_file, indent=4)
        except FileNotFoundError:
            self.logger.error(f"Error: The file path '{json_result_full_path}' does not exist.")
        except PermissionError:
            self.logger.error(f"Error: You do not have permission to write to '{json_result_full_path}'.")
        except json.JSONDecodeError:
            self.logger.error("Error: Failed to serialize results to JSON.")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
        else:
            self.logger.info(f"Saved results into {json_result_full_path}.")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process the frame into grayscale to run Haar cascade.
        Apply histogram equalization if the contrast is too low.
        """
        preprocessed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(preprocessed_frame) < 50:
            preprocessed_frame = cv2.equalizeHist(preprocessed_frame)
        return preprocessed_frame

    def _detect_on_frame(self, frame_gray: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Apply all the enabled Haar cascade(s) on the given frame and return all the detections.

        Args:
            frame_gray (np.ndarray): The frame ready to be used for detection - expected to be grayscale

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary with
                - Key: The name of the cascade
                - Value: a List (each entry being a detection done on the frame)
                           containing dictionary with the detection data (detections, reject_levels, weights)
        """
        detections_by_cascades_frame: Dict[str, List[Dict[str, Any]]] \
            = {config["name"]: [] for config in self.cascades_config}

        for name, cascade in self.cascades.items():
            detections, reject_levels, level_weights = self._detect_faces(
                cascade['classifier'],
                frame_gray,
                cascade["scale_factor"],
                cascade["min_neighbors"],
                tuple(cascade["min_size"])
            )

            detections_by_cascades_frame[name].append(
                {
                    "detections": detections.tolist() if len(detections) > 0 else [],
                    "reject_levels": list(reject_levels) if reject_levels is not None else [],
                    "weights": list(level_weights) if level_weights is not None else []
                })
        return detections_by_cascades_frame

    def _detect_faces(self,
                      cascade: cv2.CascadeClassifier,
                      frame_gray: np.ndarray,
                      scale_factor: float,
                      min_neighbors: int,
                      min_size: Tuple[int, int]):
        """
        Apply a CascadeClassifier on the given grayscale frame and run the detection.

        Args:
            cascade (cv2.CascadeClassifier): Haar cascade to use for the detection.
            frame_gray (np.ndarray): Grayscale to perform the detection on.
            scale_factor (float): scale_factor parameter for detectMultiScale3 (cf. doc) - tldr. More = + results
            min_neighbors (int): min_neighbors parameter for detectMultiScale3 (cf. doc) - tldr. More = + accurate
            min_size (Tuple[int, int]): Minimum object size to try to detect.

        Returns:
            detections, reject_levels, level_weights
        """
        assert len(frame_gray) > 0
        assert scale_factor > 0
        assert min_neighbors > 0
        try:
            detections, reject_levels, level_weights = cascade.detectMultiScale3(
                frame_gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                outputRejectLevels=True
            )
        except cv2.error as e:
            self.logger.error(f"Error with detectMultiScale3: {e}")
            exit(1)
        return detections, reject_levels, level_weights

    def _load_detections(self):
        """
        Load the matching detection file if it exists, and set it into self.detections_by_cascades.

        Returns:
            bool: True if the loading succeeded, False otherwise.
        """
        json_result_full_path: str = self.general_config["output_json_dir"]+self.general_config["output_json_name"]
        if os.path.exists(json_result_full_path):
            self.logger.info(f"Results file '{json_result_full_path}' already exists. Loading results from file.")
            try:
                with open(json_result_full_path, "r") as json_file:
                    self.all_detections_by_cascades = json.load(json_file)
                    return True
            except json.JSONDecodeError as e:
                self.logger.error(f"Error: JSON results format is invalid. Change file or fix format. {e}."
                                  f" Exiting program.")
                exit(1)
        else:
            self.logger.info(f"No results file existing at '{json_result_full_path}'. Performing detection.")

    def convert_to_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(element) for element in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _filter_by_weight_frame(self, detections_frame: Dict):
        """
        Read the list of detection by cascade, remove the one that doesn't match the threshold and return the new list.
        """
        filtered_results = {}
        for cascade_name, detections in detections_frame.items():
            weight_threshold = next(
                config["confidence_threshold"] for config in self.cascades_config if config["name"] == cascade_name
            )
            filtered_results[cascade_name] = [
                detection for detection in detections
                if detection["weights"] and max(detection["weights"]) >= weight_threshold
            ]
        return filtered_results

    def _filter_centers_frame(self, detections: Dict) -> List[Dict]:
        """
        Merge all detections closer than a specified distance threshold.
        @AI generated method

        Args:
            detections (Dict): Detections grouped by cascade names.

        Returns:
            List[Dict]: A list of merged detections.
        """
        distance_threshold = self.general_config.get("distance_between_centers", 100)
        merged_detections = []

        # Not ideal to use 3 loops. Look for optimization?
        for cascade_name, cascade_detections in detections.items():
            for detection_entry in cascade_detections:
                for bounding_box in detection_entry["detections"]:
                    x, y, w, h = bounding_box
                    center = (x + w // 2, y + h // 2)

                    is_merged = False
                    for merged in merged_detections:
                        mx, my, mw, mh = merged["detections"]
                        merged_center = (mx + mw // 2, my + mh // 2)
                        distance = np.sqrt((center[0] - merged_center[0])**2 + (center[1] - merged_center[1])**2)

                        if distance < distance_threshold:
                            # Merge detections by expanding the bounding box
                            merged["detections"] = [
                                min(x, mx),
                                min(y, my),
                                max(x + w, mx + mw) - min(x, mx),
                                max(y + h, my + mh) - min(y, my)
                            ]
                            is_merged = True
                            break

                    if not is_merged:
                        merged_detections.append({"detections": [x, y, w, h]})

        return merged_detections

    def _blur_frame(self, frame, blur_zones: List[Dict]) -> np.ndarray:
        """
        Blur the frame based on the List in parameters.

        Args:
            frame (np.ndarray): frame to blur
            blur_zones (List[Dict]): list of the detections that should be blurred.
        """
        for zone in blur_zones:
            x, y, w, h = zone["detections"]
            roi = frame[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
            frame[y:y+h, x:x+w] = blurred_roi
        return frame

    def run(self):
        """
        Start the loading/init/detection process.
        """
        if not self._load_detections():
            self._prepare_cascades()
            self._detect_on_video()
            # Reset video to frame 0, so it's ready to be processed after detection.
            self.source_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self._detect_on_video()


if __name__ == '__main__':
    f = FaceDetector(config_file="config2.json")
    f.run()
    # @todo Better separation of detection/saving results and video process.
