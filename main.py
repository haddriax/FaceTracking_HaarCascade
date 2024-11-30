import cv2
import cv2.cuda
import math
import numpy as np


def remove_close_centers_old(centers, min_distance):
    centers = sorted(centers, key=lambda c: c[0] + c[1])
    filtered_centers = []
    for center in centers:
        if all(math.dist(center, existing) >= min_distance for existing in filtered_centers):
            filtered_centers.append(center)

    return filtered_centers


def remove_close_centers(centers, min_distance):
    centers = np.array(sorted(centers, key=lambda c: c[0] + c[1]))
    filtered_centers = []
    for center in centers:
        if all(np.linalg.norm(center[:2] - np.array(existing[:2])) >= min_distance for existing in filtered_centers):
            filtered_centers.append(center)
    return filtered_centers


def draw_rectangles(frame, objects, weights, color: (float, float, float), confidence_threshold=0.0):
    for (x, y, w, h), confidence in zip(objects, weights):
        if confidence - confidence_threshold >= 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            score_text = f"{confidence:.2f}"
            cv2.putText(frame,
                        score_text,
                        (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        color,
                        3)


def execute_detection(cascades: list):
    pass


def get_centers_list(all_detections):
    centers = [(x + w // 2, y + h // 2, w, h) for (x, y, w, h) in all_detections]
    return centers


def filter_centers(centers, distance_threshold=120):
    return remove_close_centers(centers, distance_threshold)


def process_video(video: cv2.VideoCapture):
    front_face_cascade = cv2.CascadeClassifier()
    front_face_cascade_alt = cv2.CascadeClassifier()
    front_face_cascade_alt2 = cv2.CascadeClassifier()
    side_face_cascade = cv2.CascadeClassifier()

    if not front_face_cascade.load(cv2.samples.findFile('cascade/haarcascade_frontalface_default.xml')):
        print('--(!)Error loading front face cascade')
        exit(0)
    if not front_face_cascade_alt.load(cv2.samples.findFile('cascade/haarcascade_frontalface_alt.xml')):
        print('--(!)Error loading alt front face cascade')
        exit(0)
    if not front_face_cascade_alt2.load(cv2.samples.findFile('cascade/haarcascade_frontalface_alt2.xml')):
        print('--(!)Error loading alt front face cascade')
        exit(0)
    if not side_face_cascade.load(cv2.samples.findFile('cascade/haarcascade_profileface.xml')):
        print('--(!)Error loading side face cascade')
        exit(0)

    fps, total_num_frames, frame_width, frame_height = get_infos(video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_vid_writer = cv2.VideoWriter('processed.mp4', fourcc, fps, (frame_width, frame_height))

    min_face_detect_size = (220, 220)

    log_average_detections = []
    for i in range(total_num_frames):
        ret, frame = video.read()

        print(f"Processing frame {i + 1}/{total_num_frames}.")
        if frame is None or not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Enhances frames with low brightness
        if np.mean(frame_gray) < 50:
            frame_gray = cv2.equalizeHist(frame_gray)
        frame_gray = cv2.bilateralFilter(frame_gray, d=9, sigmaColor=75, sigmaSpace=75)

        processed_frame = frame

        blue = False
        green = True
        red = True
        purple = True

        faces = []

        # BLUE - Good detection but often a lot of false positive with this cascade.
        if blue:
            faces_1, rejects_level_1, level_weights_1 = detect_face(
                frame_gray,
                front_face_cascade,
                1.012,
                28,
                min_face_detect_size)
            faces.append(faces_1)
            # draw_rectangles(processed_frame, faces_1, level_weights_1, (255, 0, 0), 2)

        # GREEN
        if green:
            faces_2, rejects_level_2, level_weights_2 = detect_face(
                frame_gray,
                front_face_cascade_alt,
                1.012,
                16,
                min_face_detect_size)
            faces.append(faces_2)
            # draw_rectangles(processed_frame, faces_2, level_weights_2, (0, 255, 0), 108.0)

        # Red - Cut at 55.2?
        if red:
            faces_3, rejects_level_3, level_weights_3 = detect_face(
                frame_gray,
                front_face_cascade_alt2,
                1.015,
                18,
                min_face_detect_size)
            faces.append(faces_3)
            # draw_rectangles(processed_frame, faces_3, level_weights_3, (0, 0, 255), 55.2)

        # PURPLE - Cut at 2.2?
        if purple:
            faces_4, rejects_level_4, level_weights_4 = detect_face(
                frame_gray,
                side_face_cascade,
                1.025,
                28,
                min_face_detect_size)
            faces.append(faces_4)
            # draw_rectangles(processed_frame, faces_4, level_weights_4, (255, 0, 255), 2.2)

        all_detections = []
        for sub in faces:
            for e in sub:
                all_detections.append(e)

        centers = get_centers_list(all_detections=all_detections)
        filtered_centers = filter_centers(centers, min_face_detect_size[0])

        for (x, y, w, h) in filtered_centers:
            # Compute the radius of the blur area
            radius = int(math.sqrt((w // 2) ** 2 + (h // 2) ** 2))

            # Define the region to blur
            top_left = (max(0, x - radius), max(0, y - radius))
            bottom_right = (min(frame_width, x + radius), min(frame_height, y + radius))

            # Create a mask for the circular area to blur
            mask = np.zeros_like(processed_frame)  # Mask with the same size as the frame
            cv2.circle(mask, (x, y), radius, (255, 255, 255), thickness=-1)  # White circle for the blur region

            # Create the blurred version of the frame
            blurred_frame = cv2.GaussianBlur(processed_frame, (25, 25), 0)

            # Extract the region that needs to be blurred using the mask
            blurred_region = cv2.bitwise_and(blurred_frame, mask)

            # Extract the original region (outside the blur) using the inverted mask
            original_region = cv2.bitwise_and(processed_frame, cv2.bitwise_not(mask))

            # Combine the blurred and original regions
            processed_frame = cv2.add(blurred_region, original_region)

            # Optionally, you can draw a circle around the center for visualization
            cv2.circle(processed_frame, (x, y), radius, (255, 0, 0), 2)
            label = f"Face: {w}x{h}"
            cv2.putText(processed_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        new_vid_writer.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video.release()
    new_vid_writer.release()


def load() -> cv2.VideoCapture:
    vid = cv2.VideoCapture("video.mp4")
    return vid


def get_infos(video: cv2.VideoCapture) -> (int, int, int, int):
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, total_num_frames, frame_width, frame_height


def detect_face(frame_gray, cascade, s, n, m_s):
    try:
        faces, reject_levels, level_weights = cascade.detectMultiScale3(
            frame_gray,
            scaleFactor=s,
            minNeighbors=n,
            minSize=m_s,
            maxSize=(450, 450),
            outputRejectLevels=True)
    except cv2.error as e:
        print(f"Error with detectMultiScale3: {e}")
        faces, reject_levels, level_weights = cascade.detectMultiScale(
            frame_gray,
            scaleFactor=s,
            minNeighbors=n,
            minSize=m_s,
            maxSize=(450, 450)
        ), None, None
    return faces, reject_levels, level_weights


if __name__ == '__main__':
    loaded_video: cv2.VideoCapture = load()
    process_video(loaded_video)
