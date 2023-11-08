
# Performer Video Recognition (stashapp/stash)

  

Docker image for a face recognition service for adult performers. The service was developed for the [stashapp/stash](https://github.com/stashapp/stash) project. The image provides an api endpoint that accepts a video file path, which is scanned using the [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) package. Candidate faces are ranked by confidence of a match to a specific performer.

  

## Get started

  

### Build the Docker image

  

Start by building the docker image with a defined name. Choose the gpu Dockerfile, if you have a compatible device.

  

```bash

docker  build  -t  stash_performer_recognition  .

```

  

### Run the Docker image

  

Start the image on a forwarded port 8080 and bind a local directory to the container's `/data` folder, which acts the source for the video path. Enable the nvidia runtime for the gpu image.

  

```bash

docker  run  -d  [--runtime=nvidia  --gpus  all]  -p8080:8080  -v  /local/path/to/videos:/data  stash_performer_recognition

```

  

## Functions

  

### Identify faces in video

  

`POST` or `GET` request to the web service with the video path.

`curl http://localhost:8080/?path=/data/path/to/video.mp4`

`curl -X POST -d "path=/data/path/to/video.mp4" http://localhost:8080/`

  

## Response Schema

  

The response is a JSON array of matching performers with the following structure.

  ```json
[
	{
	  "image": "base-64 encoded image of the captured face",
	  "matches":  [
			{
			  "name": "performer name",
			  "poster": "performer thumbnail image",
			  "score": "match confidence score (0-100)",
			  "urls": ["list of profile urls"], 
			},
			// ...
		],
	}
]
```

## Notes

  

This project is purely experimental, and shouldn't be used in any situation where reliability is a factor . The underlying python classes have additional settings such as frame intervals for image capture and a minimum confidence filter.  

Feel free to experiment!