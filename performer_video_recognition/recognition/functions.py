"""
Utility functions for performer_video_recognition package
"""

import base64
import cv2
import face_recognition
import numpy as np
from PIL import Image
from io import BytesIO


def base64_encode(image_path: str) -> str:
    """
    The function `base64_encode` takes an image file path as input, reads the file in binary mode,
    encodes it using base64 encoding, and returns the encoded string.

    :param image_path: The `image_path` parameter is a string that represents the file path of the image
    that you want to encode
    :type image_path: str
    :return: a string that represents the base64 encoding of the image file.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def pil_to_cv2(pil_img: Image) -> np.ndarray:
    """
    The function `pil_to_cv2` converts a PIL image object to a numpy array using OpenCV.

    :param pil_img: The pil_img parameter is of type Image, which is an image object from the PIL
    (Python Imaging Library) module
    :type pil_img: Image
    :return: a numpy array representing the image in OpenCV format.
    """
    buffer = BytesIO()
    pil_img.save(
        buffer, format="JPEG"
    )  # You can change format to PNG if you want lossless compression
    buffer.seek(0)
    image_np = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
    return cv2.imdecode(image_np, cv2.IMREAD_COLOR)
