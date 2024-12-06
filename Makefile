# Local image name
IMG_NAME=optimo
VERSION=0.1.0
IMG_REGI=javierarroyo/optimo

build-optimo:
	docker build --progress=plain --platform=linux/amd64 --rm -t ${IMG_NAME} .

build-optimo-no-cache:
	docker build --progress=plain --platform=linux/amd64 --no-cache --rm -t ${IMG_NAME} .

run-optimo: 
	docker run \
		--platform=linux/amd64 \
		--name ${IMG_NAME} \
		--detach=false \
		--network=host \
		-w /home/developer \
		--rm -it ${IMG_NAME}

push-optimo:
# requires `docker login` first
	docker tag ${IMG_NAME} ${IMG_REGI}:${VERSION}
	docker push ${IMG_REGI}:${VERSION}

pull-optimo:
	docker pull ${IMG_REGI}:${VERSION}
	docker tag ${IMG_REGI}:${VERSION} ${IMG_NAME}

test-examples:
	python3 -m unittest tests.test_examples.Test_Optimo_Examples
