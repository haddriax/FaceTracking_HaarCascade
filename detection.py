import cv2
import math
import json
import os
import numpy as np


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def remove_close_centers(centers, min_distance):
    centers = np.array(sorted(centers, key=lambda c: c[0] + c[1]))
    filtered_centers = []
    for center in centers:
        if all(np.linalg.norm(center[:2] - np.array(existing[:2])) >= min_distance for existing in filtered_centers):
            filtered_centers.append(center)
    return filtered_centers


def draw_rectangles(frame, objects, color: (float, float, float)):
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"Face: {w}x{h}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def get_centers_list(detections):
    centers = [(x + w // 2, y + h // 2, w, h) for (x, y, w, h) in detections]
    return centers


def filter_centers(centers, distance_threshold=120):
    return remove_close_centers(centers, distance_threshold)


def load_or_run_detection(video, cascades_config, processed_video_path, output_json_path):
    if os.path.exists(output_json_path):
        print(f"JSON file detected: {output_json_path}. Loading detections.")
        with open(output_json_path, "r") as json_file:
            detections_data = json.load(json_file)
        return detections_data
    else:
        print(f"No existing JSON file. Running detection and saving results to {output_json_path}.")
        detections_data = run_detection(video, cascades_config)
        with open(output_json_path, "w") as json_file:
            detections_data_serializable = convert_to_serializable(detections_data)
            json.dump(detections_data_serializable, json_file, indent=4)

        return detections_data


def run_detection(video, cascades_config):
    fps, total_num_frames, frame_width, frame_height = get_infos(video)
    detections_by_cascade = {config["name"]: [] for config in cascades_config}

    for i in range(total_num_frames):
        ret, frame = video.read()

        print(f"Processing frame {i + 1}/{total_num_frames}.")
        if frame is None or not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(frame_gray) < 50:
            frame_gray = cv2.equalizeHist(frame_gray)
        frame_gray = cv2.bilateralFilter(frame_gray, d=9, sigmaColor=75, sigmaSpace=75)

        for config in cascades_config:
            if config["enabled"]:
                cascade = cv2.CascadeClassifier(config["cascade_path"])
                detections, reject_levels, level_weights = detect_face(
                    frame_gray,
                    cascade,
                    config["scale_factor"],
                    config["min_neighbors"],
                    tuple(config["min_size"])
                )

                # Store results with reject levels and weights
                detections_by_cascade[config["name"]].append({
                    "detections": detections.tolist() if len(detections) > 0 else [],
                    "reject_levels": list(reject_levels) if reject_levels is not None else [],
                    "weights": list(level_weights) if level_weights is not None else []
                })

    return detections_by_cascade


def run_detection_bs(video, cascades_config):
    fps, total_num_frames, frame_width, frame_height = get_infos(video)
    detections_by_cascade = {config["name"]: [] for config in cascades_config}
    detections_by_frame = []
    cascades = prepare_cascades(cascades_config)

    for i in range(total_num_frames):
        ret, frame = video.read()

        detections_by_frame.append([])

        print(f"Processing frame {i + 1}/{total_num_frames}.")
        if frame is None or not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(frame_gray) < 50:
            frame_gray = cv2.equalizeHist(frame_gray)
        frame_gray = cv2.bilateralFilter(frame_gray, d=9, sigmaColor=75, sigmaSpace=75)


        for config in cascades_config:
            if config["enabled"]:
                cascade = cv2.CascadeClassifier(config["cascade_path"])
                detections, reject_levels, level_weights = detect_face(
                    frame_gray,
                    cascade,
                    config["scale_factor"],
                    config["min_neighbors"],
                    tuple(config["min_size"])
                )

                # Store results with reject levels and weights
                detections_by_cascade[config["name"]].append({
                    "detections": detections.tolist() if len(detections) > 0 else [],
                    "reject_levels": list(reject_levels) if reject_levels is not None else [],
                    "weights": list(level_weights) if level_weights is not None else []
                })

    return detections_by_cascade


def prepare_cascades(cascades_config) -> dict[str: cv2.CascadeClassifier]:
    """
    Generate all the CascadeClassifier objects if they are activated in the config file
    :type cascades_config: object
    """
    return {config["cascade_path"]: cv2.CascadeClassifier(config["cascade_path"]) for config in cascades_config if
            config["enabled"]}


def run_detection_on_frame(frame_gray, config, cascades):
    for cascade in cascades:
        if config["enabled"]:
            cascade = cv2.CascadeClassifier(config["cascade_path"])
            detections, reject_levels, level_weights = detect_face(
                frame_gray,
                cascade,
                config["scale_factor"],
                config["min_neighbors"],
                tuple(config["min_size"])
            )

        return [config["name"]].append({
            "detections": detections.tolist() if len(detections) > 0 else [],
            "reject_levels": list(reject_levels) if reject_levels is not None else [],
            "weights": list(level_weights) if level_weights is not None else []
        })


def read_detection_json(detections_json, cascade_config, frame_index):
    if cascade_config["enabled"]:
        frame_detections = detections_data[cascade_config["name"]][frame_index]
        detections = frame_detections["detections"]
        weights = frame_detections.get("weights", [])

        filtered_detections_by_weight = [
            detection for detection, weight
            in zip(detections, weights)
            if weight >= cascade_config.get("confidence_threshold")
        ]

        if cascade_config["draw"]:
            pass


def process_video(video: cv2.VideoCapture, detections_data, cascades_config, processed_video_path):
    fps, total_num_frames, frame_width, frame_height = get_infos(video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_vid_writer = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

    for i in range(total_num_frames):
        ret, frame = video.read()

        print(f"Processing frame {i + 1}/{total_num_frames}.")
        if frame is None or not ret:
            break

        processed_frame = frame

        cascade_list = []
        # Read all the detections that we stored in JSON format, by cascade.
        for config in cascades_config:

            # Only apply the JSON if
            if config["enabled"]:
                frame_detections = detections_data[config["name"]][i]
                detections = frame_detections["detections"]
                weights = frame_detections.get("weights", [])

                filtered_detections_by_weight = [
                    detection for detection, weight
                    in zip(detections, weights)
                    if weight >= config.get("confidence_threshold")
                ]
                cascade_list.append(filtered_detections_by_weight)

                if config["draw"]:
                    draw_rectangles(processed_frame, filtered_detections_by_weight, config["color"])

        flattened_list = [item for sublist in cascade_list for item in sublist]
        matching_centers = filter_centers(flattened_list, 120)
        blur_frame(processed_frame, matching_centers)

        new_vid_writer.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video.release()
    new_vid_writer.release()


def detect_face(frame_gray, cascade, scale_factor, min_neighbors, min_size):
    try:
        faces, reject_levels, level_weights = cascade.detectMultiScale3(
            frame_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            outputRejectLevels=True
        )
    except cv2.error as e:
        print(f"Error with detectMultiScale3: {e}")
        faces, reject_levels, level_weights = cascade.detectMultiScale(
            frame_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        ), None, None
    return faces, reject_levels, level_weights


def load_video(video_path: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(video_path)


def get_infos(video: cv2.VideoCapture):
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, total_num_frames, frame_width, frame_height


def blur_frame(processed_frame, filtered_centers):
    for (x, y, w, h) in filtered_centers:
        radius = int(math.sqrt((w // 2) ** 2 + (h // 2) ** 2))
        mask = np.zeros_like(processed_frame)
        cv2.circle(mask, (x, y), radius, (255, 255, 255), thickness=-1)
        blurred_frame = cv2.GaussianBlur(processed_frame, (25, 25), 0)
        blurred_region = cv2.bitwise_and(blurred_frame, mask)

        original_region = cv2.bitwise_and(processed_frame, cv2.bitwise_not(mask))
        processed_frame = cv2.add(blurred_region, original_region)


if __name__ == '__main__':
    video_path = "video.mp4"
    processed_video_path = "processedV3.mp4"
    output_json_path = os.path.splitext(processed_video_path)[0] + ".json"

    cascades_config = [
        {
            "name": "frontalface_default",
            "enabled": False,
            "draw": False,
            "color": (255, 0, 0),
            "cascade_path": cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            "scale_factor": 1.012,
            "min_neighbors": 28,
            "min_size": [220, 220],
            "confidence_threshold": 2
        },
        {
            "name": "frontalface_alt",
            "enabled": True,
            "draw": True,
            "color": (0, 255, 0),
            "cascade_path": cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
            "scale_factor": 1.012,
            "min_neighbors": 16,
            "min_size": [220, 220],
            "confidence_threshold": 108.0
        },
        {
            "name": "frontalface_alt2",
            "enabled": True,
            "draw": True,
            "color": (0, 0, 255),
            "cascade_path": cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
            "scale_factor": 1.015,
            "min_neighbors": 18,
            "min_size": [220, 220],
            "confidence_threshold": 55.2
        },
        {
            "name": "profileface",
            "enabled": True,
            "draw": True,
            "color": (255, 0, 255),
            "cascade_path": cv2.data.haarcascades + 'haarcascade_profileface.xml',
            "scale_factor": 1.025,
            "min_neighbors": 28,
            "min_size": [220, 220],
            "confidence_threshold": 2.2
        }
    ]

    video = load_video(video_path)
    detections_data = load_or_run_detection(video, cascades_config, processed_video_path, output_json_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the first frame
    process_video(video, detections_data, cascades_config, processed_video_path)
