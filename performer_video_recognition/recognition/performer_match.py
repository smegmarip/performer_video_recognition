"""
Performer Match sends images to a performer facial recognition API
and returns the results.
"""
import os
import uuid
import requests
import numpy as np
from typing import List
from bs4 import BeautifulSoup
from .functions import base64_encode


class PerformerMatch:
    """
    The PerformerMatch class is used for matching performers based on certain criteria.
    """

    api_url = "https://pornstarbyface.com/Home/LooksLikeByPhoto"

    def __init__(self, temp_dir: str = "/tmp/") -> None:
        """
        The function initializes an object with a temporary directory path.

        :param temp_dir: The `temp_dir` parameter is a string that represents the directory path where
        temporary files will be stored. By default, it is set to "/tmp/", which is a common directory used
        for temporary files in Unix-like operating systems. However, you can provide a different directory
        path as an argument when creating, defaults to /tmp/
        :type temp_dir: str (optional)
        """
        self.temp_dir = temp_dir

    def match_file(self, image_path: str) -> List:
        """
        The function takes an image file path, reads the file, calls an API with the file, and returns the
        image path and the matches obtained from the API.

        :param image_path: The `image_path` parameter is a string that represents the file path of the image
        that you want to match
        :type image_path: str
        :return: a dictionary with two keys: "image" and "matches". The value of the "image" key is the
        base64 encoding of the image file path, and the value of the "matches" key is the result of calling
        the `call_api` function with the image file.
        """
        with open(image_path, "rb") as f:
            files = {"imageUploadForm": (image_path, f.read(), "image/jpeg")}

        matches = self.call_api(files)
        result = {"image": base64_encode(image_path), "matches": matches}
        return result

    def match_image(self, image: np.ndarray) -> List:
        """
        The function takes an image as input, saves it to disk, matches it with a file, removes the saved
        image, and returns the result.

        :param image: The `image` parameter is of type `np.ndarray`, which stands for NumPy array. It
        represents an image in the form of a multi-dimensional array
        :type image: np.ndarray
        :return: the result of the `self.match_file(image_path)` function call.
        """
        # Construct an image file path
        random_filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(self.temp_dir, random_filename)

        # Save the PIL Image to disk
        image.save(image_path, "JPEG")
        result = self.match_file(image_path)
        os.remove(image_path)
        return result

    def call_api(self, files: dict, limit: int = 5) -> List:
        """
        The `call_api` function sends a POST request to an API with files, parses the HTML response,
        extracts relevant information, and returns a sorted list of data with a specified limit.

        :param files: The `files` parameter is a dictionary that contains the files to be sent in the API
        request. The keys of the dictionary represent the names of the files, and the values represent the
        file objects
        :type files: dict
        :param limit: The `limit` parameter is an optional integer that specifies the maximum number of
        results to return from the API call. By default, it is set to 5, but you can provide a different
        value if needed, defaults to 5
        :type limit: int (optional)
        :return: a list of dictionaries. Each dictionary contains the name, urls, and score of a candidate.
        The list is sorted in descending order based on the score and limited to the specified limit.
        """
        data = []
        r = requests.post(self.api_url, files=files)
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        mydivs = soup.findAll("div")
        score = 0
        base_url = "https://pornstarbyface.com"
        for div in mydivs:
            urls = []
            poster = ""
            img = div.find("img", attrs={"class": "img-thumbnail"}, recursive=True)
            if img is not None:
                poster = base_url + img["src"]
            if "progress-bar" in div.get("class", []):
                score = int(div["similarity"])
            elif div.get("class", "") == ["candidate-real"]:
                for d in div.findAll("div"):
                    p = d.find("p")
                    name = p.text
                    for a in d.findAll("a"):
                        urls.append(a["href"])
                    data.append({"name": name, "urls": urls, "poster": poster, "score": score})

        return sorted(data, key=lambda x: x["score"], reverse=True)[:limit]
