PROJECT_NAME := openai-access-gateway

# VERSION is the version we should download and use.
VERSION:=$(shell git rev-parse --short HEAD)
# DOCKER is the docker image repo we need to push to.
DOCKER_REPO:=defangio
DOCKER_IMAGE_NAME:=$(DOCKER_REPO)/$(PROJECT_NAME)

DOCKER_IMAGE_ARM64:=$(DOCKER_IMAGE_NAME):arm64-$(VERSION)
DOCKER_IMAGE_AMD64:=$(DOCKER_IMAGE_NAME):amd64-$(VERSION)

DEFAULT_MODEL := default

.PHONY: no-diff
no-diff:
	git diff-index --quiet HEAD -- src     # check that there are no uncommitted changes

.PHONY: push
push: login
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--build-arg DEFAULT_MODEL=$(DEFAULT_MODEL) \
		-f ./src/Dockerfile_ecs \
		-t $(DOCKER_IMAGE_NAME):$(VERSION) \
		--push \
		./src

.PHONY: login
login: ## Login to docker
	@docker login
