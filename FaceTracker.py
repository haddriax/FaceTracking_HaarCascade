import numpy as np
from scipy.optimize import linear_sum_assignment


class FaceTracker:
    def __init__(self, max_before_untrack=10, distance_threshold=50):
        """
        Initialize the tracker.

        Args:
            max_before_untrack (int): Maximum number of frames a face can go undetected before being removed.
            distance_threshold (float): Maximum allowable distance for assignment.
        """
        self.next_track_id = 0
        self.tracks = {}  # Stores active tracks
        self.tracking_paths = {}
        self.max_before_untrack = max_before_untrack
        self.distance_threshold = distance_threshold

    def _calculate_cost_matrix(self, current_detections):
        """
        Compute the cost matrix between current detections and active tracks.

        Args:
            current_detections (List[List[int]]): Current frame's detections (list of [x, y, w, h]).

        Returns:
            np.ndarray: Cost matrix.
        """
        current_centers = [((x + w // 2), (y + h // 2)) for x, y, w, h in current_detections]
        track_centers = [
            ((track["bbox"][0] + track["bbox"][2] // 2), (track["bbox"][1] + track["bbox"][3] // 2))
            for track in self.tracks.values()
        ]

        cost_matrix = np.zeros((len(current_centers), len(track_centers)), dtype=np.float32)
        for i, det_center in enumerate(current_centers):
            for j, track_center in enumerate(track_centers):
                cost_matrix[i, j] = np.linalg.norm(np.array(det_center) - np.array(track_center))

        return cost_matrix

    def update(self, detections):
        """
        Update tracks with current detections.

        Args:
            detections (List[List[int]]): Current frame's detections (list of [x, y, w, h]).

        Returns:
            Dict: Updated tracks.
        """
        if len(self.tracks) == 0:
            # No active tracks? initialize new tracks
            for detection in detections:
                self.tracks[self.next_track_id] = {"bbox": detection, "disappeared": 0}
                self.next_track_id += 1
        else:
            cost_matrix = self._calculate_cost_matrix(detections)

            # Hungarian method
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            # Track unmatched detections and tracks
            unmatched_detections = set(range(len(detections)))
            unmatched_tracks = set(self.tracks.keys())

            # Process matches
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] > self.distance_threshold:
                    continue  # Skip assignments that are too far

                track_id = list(self.tracks.keys())[col]
                self.tracks[track_id]["bbox"] = detections[row]
                self.tracks[track_id]["disappeared"] = 0

                unmatched_detections.discard(row)
                unmatched_tracks.discard(track_id)

            # Handle unmatched detections: create new tracks
            for row in unmatched_detections:
                self.tracks[self.next_track_id] = {"bbox": detections[row], "disappeared": 0}
                self.next_track_id += 1

            # Handle unmatched tracks: mark as disappeared
            for track_id in unmatched_tracks:
                self.tracks[track_id]["disappeared"] += 1
                if self.tracks[track_id]["disappeared"] > self.max_before_untrack:
                    del self.tracks[track_id]

        return self.tracks
