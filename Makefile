# Local image name
IMG_NAME=optimo

build:
	docker build --progress=plain --platform=linux/amd64 --rm -t ${IMG_NAME} .

build-no-cache:
	docker build --progress=plain --platform=linux/amd64 --no-cache --rm -t ${IMG_NAME} .

run: 
	docker run \
		--platform=linux/amd64 \
		--name modopti \
		--detach=false \
		--network=host \
		-w /home/developer \
		--rm -it modopti
