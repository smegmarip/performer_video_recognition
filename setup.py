"""Setup Config"""
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="performer_video_recognition",
    version="0.0.1",
    packages=[
        "performer_video_recognition",
    ],
    install_requires=[
        "beautifulsoup4>=4.12.2",
        "opencv-python>=4.0.0",
        "dlib>=19.7",
        "face-recognition>=1.3.0",
        "tqdm>=4.0.0",
        "flask>=3.0.0",
        "flask_cors>= 4.0.0",
        "pathlib>=1.0.0",
        "scikit-build>=0.17.6",
	"requests>=2.0.0",
    ],
    license="GPL",
    url="https://github.com/smegmarip/",
    author="smegmarip",
    author_email="smegmarip@gmail.com",
    description="API for video scraping and performer identification",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
