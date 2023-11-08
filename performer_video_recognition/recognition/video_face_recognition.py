"""
Video Face Recognition Class
Scans video at intervals and identifies faces in captured frames.
"""
import os
import argparse
import uuid
from typing import List, Tuple, Any
from collections import defaultdict
from pprint import pprint
import numpy as np
import cv2
import dlib
import face_recognition
from tqdm import tqdm
from PIL import Image
from .performer_match import PerformerMatch
from .progressbar import ProgressBar

# DLIB Landmarks data
script_dir = os.path.dirname(os.path.abspath(__file__))
landmarks_file = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")

# DLIB sub-detectors
DLIB_SUBD = ["n/a", "front", "left", "right", "front-rotate-left", "front-rotate-right"]


class VideoFaceRecognition:
    """
    The `VideoFaceRecognition` class is a Python class that performs face recognition on a video file,
    detects faces in each frame, crops the frames around the faces, normalizes the orientation of the
    faces, calculates image quality scores for each face, groups similar faces together, and saves the
    processed faces to individual directories.
    """

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor(landmarks_file)

    face_frames = defaultdict(dict)

    face_locations = defaultdict(dict)

    results = []

    progress_bar = None

    def __init__(
        self,
        interval: int = 5,
        output_dir: str = "~",
        algorithm: str = "dlib",
        padding: int = 45,
        display_progress: bool = False,
    ) -> None:
        """
        The function initializes an object with various parameters for a face recognition algorithm.

        :param interval: The interval parameter determines the number of frames between each face
        detection, defaults to 5
        :type interval: int (optional)
        :param output_dir: The `output_dir` parameter is a string that specifies the directory where the
        output files will be saved. By default, it is set to `"~"`, which represents the user's home
        directory. However, the `os.path.expanduser()` function is used to expand the `~` character to,
        defaults to ~
        :type output_dir: str (optional)
        :param algorithm: The "algorithm" parameter specifies the face detection algorithm to be used.
        The default value is set to "dlib", which refers to the dlib library's face detection algorithm,
        defaults to dlib
        :type algorithm: str (optional)
        :param padding: The `padding` parameter is an integer that represents the number of pixels to
        add around the detected face in the output image. This is useful to include some background
        around the face in the output image, defaults to 45
        :type padding: int (optional)
        :param display_progress: The `display_progress` parameter is a boolean flag that determines
        whether or not to display progress information during the execution of the code. If set to
        `True`, progress information will be displayed. If set to `False`, progress information will not
        be displayed, defaults to False
        :type display_progress: bool (optional)
        """
        self.interval = interval
        self.quality_mode = f"{algorithm}_quality"
        self.output_dir = os.path.expanduser(output_dir)
        self.padding = padding
        self.display_progress = display_progress

    def capture(self, video_file: str) -> None:
        """
        The `capture` function captures frames from a video file at a specified interval, detects faces
        in each frame using face_recognition library, and stores the frames with detected faces along
        with their bounding box coordinates.

        :param video_file: The video_file parameter is a string that represents the file path of the
        video that you want to capture frames from
        :type video_file: str
        """
        # Load the video file
        video_capture = cv2.VideoCapture(video_file)

        # Get the video's frame rate
        frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))

        # Calculate the number of frames to skip for the given interval
        frames_to_skip = int(frame_rate * self.interval)

        # Get the total number of frames in the video
        if self.display_progress:
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_bar = ProgressBar(total_frames)

        # Initialize variables
        frame_count = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Increment frame count
            frame_count += 1

            # Skip frames based on the defined interval
            if frame_count % frames_to_skip != 0:
                if self.display_progress:
                    self.progress_bar.update_progressbar()
                continue

            # Convert the frame to RGB (required for face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find face locations in the frame
            face_locations = face_recognition.face_locations(rgb_frame)

            if len(face_locations) > 0:
                self.face_frames[frame_count] = {"frame": frame, "bbox": face_locations}

            if self.display_progress:
                self.progress_bar.update_progressbar()

        # Release video capture
        video_capture.release()

        # Close the tqdm progress bar
        if self.display_progress:
            self.progress_bar.close_progressbar()

    def crop_frames(self) -> None:
        """
        The function `crop_frames` loops through each detected face in a frame, crops the frame around
        the face, and stores the cropped faces along with their average pixel brightness.
        """
        # Loop through each detected face and crop the frame around the face
        for idx, frame_data in self.face_frames.items():
            frame = frame_data["frame"]
            custom_faces = self.custom_normalize_faces(frame_data)
            dlib_faces = []

            # Detect and normalize whole frame with dlib (no dark background)
            normalized_faces = self.dlib_normalize_faces(frame)
            if len(normalized_faces):
                for face in normalized_faces:
                    # Calculate the average pixel value for the face region
                    avg_pixel_value = (
                        128 if 0 in face["tight"].shape else np.mean(face["tight"])
                    )

                    dlib_faces.append(
                        {
                            "cropped": face["loose"],
                            "tight": face["tight"],
                            "brightness": avg_pixel_value,
                        }
                    )

            # Choose dlib if successful, else face recog
            self.face_frames[idx]["faces"] = (
                dlib_faces if len(dlib_faces) else custom_faces
            )

    def custom_normalize_faces(self, frame_data: dict) -> List:
        """
        The function `custom_normalize_faces` takes in frame data containing bounding box coordinates of
        faces, crops the frame around each face with padding, normalizes the orientation of the face,
        calculates the average pixel value for the face region, and returns the cropped face data along
        with the brightness value.

        :param frame_data: The `frame_data` parameter is a dictionary that contains information about
        the frame and the bounding boxes of the faces detected in the frame. It has the following
        structure:
        :type frame_data: dict
        :return: a list of dictionaries, where each dictionary represents a face in the frame. Each
        dictionary contains the following keys:
        - "cropped": the cropped face image with padding
        - "tight": the cropped face image without padding
        - "brightness": the average pixel value for the face region
        """
        frame = frame_data["frame"]
        face_data = []

        for i, (top, right, bottom, left) in enumerate(frame_data["bbox"]):
            # Calculate the padding for each side while ensuring it doesn't go beyond the image edge
            top_pad = min(self.padding, top)
            right_pad = min(self.padding, frame.shape[1] - right)
            bottom_pad = min(self.padding, frame.shape[0] - bottom)
            left_pad = min(self.padding, left)

            # Crop the frame around the face with padding
            face_top = max(0, top - top_pad)
            face_right = min(frame.shape[1], right + right_pad)
            face_bottom = min(frame.shape[0], bottom + bottom_pad)
            face_left = max(0, left - left_pad)

            # Crop the frame with padding
            tight_frame = frame[top:bottom, left:right]
            loose_frame = frame[face_top:face_bottom, face_left:face_right]

            # Normalize orientation
            normalized_faces = self.dlib_normalize_faces(loose_frame)
            face = (
                normalized_faces[0]
                if len(normalized_faces)
                else {"loose": loose_frame, "tight": tight_frame}
            )

            # Calculate the average pixel value for the face region
            avg_pixel_value = (
                128 if 0 in face["tight"].shape else np.mean(face["tight"])
            )

            # Store the face frame
            face_data.append(
                {
                    "cropped": face["loose"],
                    "tight": face["tight"],
                    "brightness": avg_pixel_value,
                }
            )
        return face_data

    def dlib_normalize_faces(self, image: np.ndarray) -> List:
        """
        The function `dlib_normalize_faces` takes an image as input, detects faces in the image, applies
        normalization techniques to the faces, and returns a list of normalized face images.

        :param image: The `image` parameter is a NumPy array representing an input image. It is expected
        to be in BGR color format
        :type image: np.ndarray
        :return: a list of dictionaries, where each dictionary contains two keys: "loose" and "tight".
        The values associated with these keys are the normalized images of the detected faces. The
        "loose" image is the face with padding applied to the bounding box, while the "tight" image is
        the face without any padding.
        """
        # Convert the image to grayscale for facial landmarks detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = self.detector(gray_image)

        normalized_images = []

        if faces and len(faces):
            for face in faces:
                # Detect facial landmarks
                landmarks = self.predictor(gray_image, face)

                # Create an affine transformation matrix for rotation and translation
                matrix = self.calc_normalized_matrix(landmarks)

                # Apply the rotation and translation to the image
                transformed_image = cv2.warpAffine(
                    image,
                    matrix,
                    (image.shape[1], image.shape[0]),
                    flags=cv2.INTER_LINEAR,
                )

                # Calculate rotated landmarks as a NumPy array
                landmarks_array = np.array(
                    [[p.x, p.y] for p in landmarks.parts()], dtype=np.float32
                )
                rotated_landmarks = cv2.transform(
                    landmarks_array.reshape(1, -1, 2), matrix
                ).reshape(-1, 2)

                # Calculate the bounding rectangle around the rotated landmarks
                x, y, w, h = cv2.boundingRect(rotated_landmarks.astype(np.int32))

                # Apply padding to the bounding box
                px = max(0, x - self.padding)
                py = max(0, y - self.padding)
                pw = min(w + (2 * self.padding), transformed_image.shape[1] - x)
                ph = min(h + (2 * self.padding), transformed_image.shape[0] - y)

                normalized_images.append(
                    {
                        "loose": transformed_image[py : py + ph, px : px + pw],
                        "tight": transformed_image[y : y + h, x : x + w],
                    }
                )

        return normalized_images

    def calc_normalized_matrix(self, landmarks: Any) -> np.ndarray:
        """
        The function calculates a normalized matrix for facial landmarks by extracting specific points
        and calculating the mean positions, angle, and center point for rotation and translation.

        :param landmarks: The "landmarks" parameter is expected to be an object that contains the facial
        landmarks detected in an image. These landmarks represent specific points on the face, such as
        the corners of the eyes, the tip of the nose, and the corners of the mouth. The landmarks are
        typically represented as a collection
        :type landmarks: Any
        :return: an affine transformation matrix that can be used for rotation and translation.
        """
        # Extract landmark points for the left eye (indices 36 to 41)
        left_eye_points = [landmarks.part(i) for i in range(36, 42)]

        # Extract landmark points for the right eye (indices 42 to 47)
        right_eye_points = [landmarks.part(i) for i in range(42, 48)]

        # Extract landmark points for the nose (indices 27 to 35)
        nose_points = [landmarks.part(i) for i in range(27, 36)]

        # Extract landmark points for the mouth (indices 48 to 67)
        mouth_points = [landmarks.part(i) for i in range(48, 68)]

        # Calculate the mean positions for each group of landmarks
        mean_left_eye = np.mean(np.array([(p.x, p.y) for p in left_eye_points]), axis=0)
        mean_right_eye = np.mean(
            np.array([(p.x, p.y) for p in right_eye_points]), axis=0
        )
        mean_nose = np.mean(np.array([(p.x, p.y) for p in nose_points]), axis=0)
        mean_mouth = np.mean(np.array([(p.x, p.y) for p in mouth_points]), axis=0)

        # Calculate the angle between the eyes (horizontal rotation)
        angle = np.degrees(
            np.arctan2(
                mean_right_eye[1] - mean_left_eye[1],
                mean_right_eye[0] - mean_left_eye[0],
            )
        )

        # Calculate the center point between eyes and nose (vertical rotation)
        center_x = (mean_left_eye[0] + mean_right_eye[0] + mean_nose[0]) / 3
        center_y = (mean_left_eye[1] + mean_right_eye[1] + mean_nose[1]) / 3

        # Create an affine transformation matrix for rotation and translation
        return cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    def image_quality_score(self, cv2_img: np.ndarray) -> int:
        """
        The `image_quality_score` function calculates a score for the quality of an image based on
        factors such as brightness, clarity, and zoom.

        :param cv2_img: The `cv2_img` parameter is a numpy array representing an image in OpenCV format.
        It is expected to be a color image (BGR format) with shape (height, width, channels)
        :type cv2_img: np.ndarray
        :return: an integer value representing the image quality score.
        """
        # Calculate the ratio of the bbox to the image area
        zoom_raw = self.calc_face_ratio(cv2_img)

        # Calculate the average pixel value (brightness)
        brightness_raw = np.mean(cv2_img)

        # Convert the image to grayscale for facial landmarks detection
        gray_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

        # Calculate the clarity score (Laplacian variance)
        clarity_raw = cv2.Laplacian(gray_image, cv2.CV_64F).var()

        # Define weight factors for brightness, clarity and zoom
        brightness_weight = 0.3 if zoom_raw == 0 else 0.4
        clarity_weight = 0.7 if zoom_raw == 0 else 0.1
        zoom_weight = 0 if zoom_raw == 0 else 0.5

        brightness_limit = 255
        clarity_limit = max(100, clarity_raw)
        zoom_limit = 100

        # Combine the scores using weighted averaging
        brightness_score = (brightness_raw / brightness_limit) * brightness_weight
        clarity_score = (clarity_raw / clarity_limit) * clarity_weight
        zoom_score = (zoom_raw / zoom_limit) * zoom_weight

        scaled_score = int((brightness_score + clarity_score + zoom_score) * 100)

        # Ensure the score is within the 0-100 range
        return max(0, min(100, scaled_score))

    def calc_face_ratio(self, cv2_img: np.ndarray) -> int:
        """
        The `calc_face_ratio` function calculates the ratio of the area occupied by a detected face to
        the area of the entire image.

        :param cv2_img: The `cv2_img` parameter is a numpy array representing an image in the OpenCV
        format. It is expected to be in BGR color format
        :type cv2_img: np.ndarray
        :return: The function `calc_face_ratio` returns an integer value representing the relative scale
        of the detected face in the input image.
        """
        # Convert the image to grayscale for facial landmarks detection
        gray_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = self.detector(gray_image)

        zoom_raw = 0
        if faces and len(faces):
            # Detect facial landmarks
            landmarks = self.predictor(gray_image, faces[0])
            min_y, max_x, max_y, min_x = self.bbox_landmarks(landmarks)
            face_width = max_x - min_x
            face_height = max_y - min_y

            # Calculate the area of the bounding rectangle
            bounding_rect_area = face_width * face_height

            # Calculate the area of the entire image
            # remove padding
            w = gray_image.shape[0]
            h = gray_image.shape[1]
            image_area = (gray_image.shape[0] * gray_image.shape[1]) - (
                (2 * (self.padding * w)) + (2 * (self.padding * h))
            )

            # Return the relative scale as a percentage
            zoom_raw = max(0, min(100, (bounding_rect_area / image_area) * 100))

        return zoom_raw

    def dlib_confidence(self, cv2_img: np.ndarray) -> Tuple:
        """
        The function `dlib_confidence` takes an OpenCV image as input, converts it to RGB format, runs a
        dlib detector on the image, and returns the confidence score and type of the detected object.

        :param cv2_img: The `cv2_img` parameter is a NumPy array representing an image in the BGR color
        space
        :type cv2_img: np.ndarray
        :return: a dictionary with two keys: "score" and "type". The value associated with the "score"
        key is the confidence score of the detected object, and the value associated with the "type" key
        is the type of the detected object. If no objects are detected, the function returns None.
        """
        rgb_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        dets, scores, idx = self.detector.run(rgb_image, 1, -1)
        if len(dets):
            return {"score": scores[0], "type": DLIB_SUBD[int(idx[0])]}
        return None

    def bbox_landmarks(self, landmarks: Any) -> Tuple:
        """
        The function `bbox_landmarks` calculates the bounding box coordinates for a set of landmarks.

        :param landmarks: The `landmarks` parameter is expected to be of type `Any`, which means it can
        be any data type. However, based on the usage in the code, it is likely that `landmarks` is
        expected to be an object that has a `parts()` method. The `parts()`
        :type landmarks: Any
        :return: a tuple containing the minimum y-coordinate, maximum x-coordinate, maximum
        y-coordinate, and minimum x-coordinate of the landmarks.
        """
        # Initialize variables to hold min and max coordinates
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = -float("inf"), -float("inf")

        # Iterate through landmarks to find min and max coordinates
        for point in landmarks.parts():
            x, y = point.x, point.y
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        # Calculate the bounding box coordinates
        # face_left = min_x
        # face_top = min_y
        # face_width = max_x - min_x
        # face_height = max_y - min_y
        return (
            min_y,
            max_x,
            max_y,
            min_x,
        )

    def group_faces(self, faces_pil, face_encodings: dict) -> List:
        """
        The `group_faces` function takes in a dictionary of face encodings and a dictionary of PIL
        images, and groups the faces based on their similarity using face recognition.

        :param faces_pil: The parameter `faces_pil` is a dictionary where the keys are strings
        representing the face images and the values are the corresponding PIL (Python Imaging Library)
        image objects
        :param face_encodings: The `face_encodings` parameter is a dictionary where the keys are strings
        representing the face index and frame number (e.g., "0_1" represents the first face in the first
        frame), and the values are the corresponding face encodings. Face encodings are numerical
        representations of facial features that
        :type face_encodings: dict
        :return: a list of groups. Each group is a list of dictionaries, where each dictionary
        represents a face.
        """
        groups = []
        for key, face_encoding in face_encodings.items():
            idx, i = key.split("_")
            if face_encoding is not None:
                matched_group = None
                for group in groups:
                    # Try matching the face with the first face of each group
                    if face_recognition.compare_faces(
                        [group[0]["encoding"]], face_encoding
                    )[0]:
                        matched_group = group
                        break

                face_data = {
                    "encoding": face_encoding,
                    "image": faces_pil[key],
                    "frame_number": int(idx),
                    "face_index": int(i),
                }

                if matched_group is not None:
                    matched_group.append(face_data)
                else:
                    # Create a new group with the face
                    groups.append([face_data])

        return groups

    def get_best_images(self, min_quality: int = 0) -> List:
        """
        The function `get_best_images` takes a minimum quality threshold as input and returns a list of
        the best images based on their quality scores.

        :param min_quality: The `min_quality` parameter is an optional integer that specifies the
        minimum quality score that an image must have in order to be considered as one of the best
        images. Images with a quality score lower than `min_quality` will be filtered out and not
        included in the final result, defaults to 0
        :type min_quality: int (optional)
        :return: a list of images.
        """
        groups = []
        for group in self.results:
            for face_data in group:
                frame_number = face_data["frame_number"]
                face_index = face_data["face_index"]
                face_image = self.face_frames[frame_number]["faces"][face_index][
                    "cropped"
                ]
                dlib_stats = self.dlib_confidence(face_image)
                face_data["custom_quality"] = self.image_quality_score(face_image)
                face_data["dlib_quality"] = (
                    50 if dlib_stats is None else dlib_stats["score"]
                )
                face_data["dlib_orientation"] = (
                    "n/a" if dlib_stats is None else dlib_stats["type"]
                )

            # Sort the group by clarity score in descending order (highest clarity first)
            groups.append(
                sorted(group, key=lambda x: x[self.quality_mode], reverse=True)
            )

        self.results = [
            list(filter(lambda f: f[self.quality_mode] >= min_quality, group))
            for group in groups
        ]
        self.results = [group for group in self.results if group]
        return [group[0]["image"] for group in self.results if len(group) > 0]

    def process_faces(self) -> List:
        """
        The function processes faces in a video by cropping them, encoding them, grouping them, and
        returning the best images.
        :return: the result of calling the `get_best_images()` method.
        """
        self.crop_frames()

        face_encodings = {}
        face_pils = {}

        for idx, frame_data in self.face_frames.items():
            for i, face_data in enumerate(frame_data["faces"]):
                key = f"{idx}_{i}"
                cropped = face_data["cropped"]
                cropped = cv2.cvtColor(face_data["cropped"], cv2.COLOR_BGR2RGB)
                face_encoding = face_recognition.face_encodings(cropped)
                if len(face_encoding):
                    face_encodings[key] = face_encoding[0]
                    face_pils[key] = Image.fromarray(cropped)
                else:
                    face_encodings[f"{idx}_{i}"] = None

        self.results = self.group_faces(face_pils, face_encodings)
        return self.get_best_images()

    def save_faces(self) -> None:
        """
        The function saves detected faces as individual images with their corresponding quality, frame
        number, and face index.
        """
        # Create output subdirectories for individuals
        os.makedirs(self.output_dir, exist_ok=True)
        job_directory = os.path.join(self.output_dir, f"{uuid.uuid4()}")
        os.makedirs(job_directory, exist_ok=True)
        for idx, individual in enumerate(self.results):
            individual_directory = os.path.join(job_directory, f"individual_{idx}")
            os.makedirs(individual_directory, exist_ok=True)

            for i, face_data in enumerate(individual):
                quality = face_data[self.quality_mode]
                frame_no = face_data["frame_number"]
                face_idx = face_data["face_index"]
                frame = self.face_frames[frame_no]["faces"][face_idx]["cropped"]
                filename = os.path.join(
                    individual_directory,
                    f"quality_{quality}_frame_{frame_no}.{face_idx}.jpg",
                )
                cv2.imwrite(filename, frame)
                self.results[idx][i]["filename"] = filename

    def process(self, video_path: str) -> List:
        """
        The function processes a video by capturing frames, detecting faces, saving the faces, and
        returning the results.

        :param video_path: The video_path parameter is a string that represents the path to the video
        file that you want to process
        :type video_path: str
        :return: the `self.results` list.
        """
        if len(self.results) > 0:
            self.results = []
            self.progress_bar = None
            self.face_frames = defaultdict(dict)
            self.face_locations = defaultdict(dict)

        self.capture(video_path)
        self.process_faces()
        self.save_faces()
        """
        pprint([{
            'filename': f['filename'],
            'quality': f[self.quality_mode],
            'frame_no': f['frame_number']
        } for individual in self.results for f in individual])
        """
        return self.results


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process video file.")

    # Define a required argument to store the video file path
    parser.add_argument("video_path", type=str, help="Path to the video file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the video file path using args.video_path
    video_path = args.video_path

    job = VideoFaceRecognition(video_path)
    res = job.process()
    api = PerformerMatch()
    for individual in res:
        pprint(api.match_file(individual[0]["filename"]))


if __name__ == "__main__":
    main()
