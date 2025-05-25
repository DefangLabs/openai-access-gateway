# Project Metadata
PROJECT_NAME       := openai-access-gateway
VERSION            ?= $(shell git rev-parse --short HEAD)

# Target Registries
AWS_REGISTRY       := public.ecr.aws/your-alias
GCP_REGISTRY       := us-docker.pkg.dev/your-gcp-project/your-repo

# Fully qualified image names
AWS_IMAGE          := $(AWS_REGISTRY)/$(PROJECT_NAME)
GCP_IMAGE          := $(GCP_REGISTRY)/$(PROJECT_NAME)

# Provider-specific models
AWS_DEFAULT_MODEL  := amazon.nova-micro-v1:0
GCP_DEFAULT_MODEL  := google/models/gemini-1.5-flash-001

# Shared Dockerfile
DOCKERFILE         := ./src/Dockerfile

# Build and push to AWS Public ECR
.PHONY: buildx-push-aws
buildx-push-aws:
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--build-arg DEFAULT_MODEL=$(AWS_DEFAULT_MODEL) \
		--build-arg PROVIDER=AWS \
		-f $(DOCKERFILE) \
		-t $(AWS_IMAGE):$(VERSION) \
		-t $(AWS_IMAGE):latest \
		--push \
		./src

# Build and push to GCP Artifact Registry
.PHONY: buildx-push-gcp
buildx-push-gcp:
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--build-arg DEFAULT_MODEL=$(GCP_DEFAULT_MODEL) \
		--build-arg PROVIDER=GCP \
		-f $(DOCKERFILE) \
		-t $(GCP_IMAGE):$(VERSION) \
		-t $(GCP_IMAGE):latest \
		--push \
		./src

# Push both AWS and GCP images (with login)
.PHONY: push-images
push-images: login-aws login-gcp buildx-push-aws buildx-push-gcp

# Ensure working directory is clean before pushing
.PHONY: no-diff
no-diff:
	git diff-index --quiet HEAD -- # fails if there are uncommitted changes

# Combined push with diff check
.PHONY: push
push: no-diff push-images

# Docker Registry Logins
.PHONY: login-aws
login-aws:
	aws ecr-public get-login-password --region us-east-1 | \
	docker login --username AWS --password-stdin public.ecr.aws

.PHONY: login-gcp
login-gcp:
	gcloud auth configure-docker us-docker.pkg.dev --quiet
