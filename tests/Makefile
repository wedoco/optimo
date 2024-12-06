# Set $ROOT to top-level directory of the repository
ROOT ?= $(shell dirname \
  $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST)))))

IMG_NAME=optimo
IMG_REGI=javierarroyo/optimo
IMG_HOME=/home/developer/optimo

VERSION=0.1.0

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
		-v ${ROOT}:${IMG_HOME}:rw \
		-w ${IMG_HOME}/tests \
		--rm -it ${IMG_NAME}

run-optimo-detached: 
	docker run \
		--platform=linux/amd64 \
		--name ${IMG_NAME} \
		--detach=true \
		--network=host \
		-v ${ROOT}:${IMG_HOME}:rw \
		-w ${IMG_HOME}/tests \
		--rm -it ${IMG_NAME}

stop-optimo:
	docker stop ${IMG_NAME}

push-optimo:
# requires `docker login` first
	docker tag ${IMG_NAME} ${IMG_REGI}:${VERSION}
	docker push ${IMG_REGI}:${VERSION}

pull-optimo:
	docker pull ${IMG_REGI}:${VERSION}
	docker tag ${IMG_REGI}:${VERSION} ${IMG_NAME}

test-examples:
	poetry config virtualenvs.in-project false && \
	poetry install && \
	poetry run python3 -m unittest test_examples.Test_Optimo_Examples

exec-optimo:
	docker exec \
		-i \
		${IMG_NAME} \
		/bin/bash -c "${ARGS} && exit"

test-examples-in-container:
	make run-optimo-detached
	make exec-optimo ARGS="make test-examples"
	make stop-optimo