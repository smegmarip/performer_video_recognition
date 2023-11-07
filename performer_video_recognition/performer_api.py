from typing import List
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
from recognition.video_face_recognition import VideoFaceRecognition
from recognition.performer_match import PerformerMatch

# Create flask app
app = Flask(__name__)
CORS(app)


def is_video(filename: str) -> bool:
    """
    The function `is_video` checks if a given filename has a video extension.

    :param filename: The `filename` parameter is a string that represents the name of a file
    :type filename: str
    :return: a boolean value, indicating whether the given filename represents a video file or not.
    """
    video_extensions = {
        "mp4",
        "mkv",
        "avi",
        "mov",
        "wmv",
        "webm",
        "flv",
        "3gp",
        "ogg",
        "mpg",
        "mpeg",
        "vob",
    }
    return "." in filename and filename.rsplit(".", 1)[1].lower() in video_extensions


def get_faces(video_path: str) -> List:
    """
    The function `get_faces` takes a video path as input and returns a list of faces detected in the
    video.

    :param video_path: The video_path parameter is a string that represents the path to the video file
    that you want to process for face recognition
    :type video_path: str
    :return: a list of faces detected in the video.
    """
    job = VideoFaceRecognition(output_dir="~/faces")
    return job.process()


def get_performers(face_groups: List):
    """
    The function "get_performers" takes a list of face groups, matches each individual face to a
    performer, and returns the results.

    :param face_groups: The parameter `face_groups` is expected to be a list of dictionaries. Each
    dictionary represents a group of faces and contains the following keys:
    :type face_groups: List
    :return: a JSON response containing the results of matching performers based on the face groups
    provided.
    """
    results = []
    if len(face_groups):
        matcher = PerformerMatch()
        for individual in face_groups:
            results.append(matcher.match_file(individual[0]["filename"]))
        return jsonify(results)
    else:
        raise BadRequest("No matches found.")


@app.route("/", methods=["POST"])
def scan():
    """
    The `scan` function takes a video path as input, checks if it is a valid video file, and then
    processes the video to detect faces and extract performer information.
    :return: the result of calling the `get_performers` function with the result of calling the
    `get_faces` function with the `video_path` argument.
    """
    # Check if a valid path was sent
    if "path" not in request.args:
        raise BadRequest("Missing file parameter: path")

    video_path = request.args["path"]
    if video_path == "" or not Path(video_path).exists():
        raise BadRequest("Given video is missing")

    if not is_video(Path(video_path).name):
        raise BadRequest("Given video has an invalid extension")

    return get_performers(get_faces(video_path))


if __name__ == "__main__":
    # Start app
    print("Starting WebServer...")
    app.run(host="0.0.0.0", port=8080, debug=False)
