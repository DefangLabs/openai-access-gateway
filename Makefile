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
push: no-diff login
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--build-arg DEFAULT_MODEL=$(DEFAULT_MODEL) \
		-f ./src/Dockerfile_ecs \
		-t $(DOCKER_IMAGE_NAME):$(VERSION) \
		-t $(DOCKER_IMAGE_NAME):latest \
		--push \
		./src

.PHONY: login
login: ## Login to docker
	@docker login

.PHONY: tests
tests:
	PYTHONPATH=src pytest

.PHONY: lint
lint: # Run pre-commit on staged/changed files
	pre-commit run

.PHONY: check
check: # Run all pre-commit hooks on all files (useful for CI or full check)
	pre-commit run --all-files

.PHONY: format
format: # Manually run ruff formatter on all files
	ruff format .

.PHONY: pre-commit-install
pre-commit-install: # Install pre-commit hooks changes
	pre-commit install
