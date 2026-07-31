# Makefile for building SHOCCS devcontainer images.
#
# Wraps Docker build commands for the two stages of .devcontainer/Dockerfile
# that are meant to be built directly:
#
#   final  (default)  the full devcontainer — toolchain + spack deps + shell,
#                     editor and Claude Code conveniences
#   ci                the lean image published to GHCR and consumed by the cpp
#                     job in .github/workflows/ci.yml
#
# The dependency tree is always built with GCC. That is fixed by the
# `packages: all: require:` line in .devcontainer/spack.yaml, not by anything
# here. (Earlier revisions of this file offered gcc/clang flavours via a
# SPACK_COMPILER build arg, but the Dockerfile never declared that ARG, so
# Docker discarded it with a warning and both flavours produced identical
# images. The targets are gone rather than left lying as a trap; building the
# TPLs with clang means adding `ARG SPACK_COMPILER` to the spack-builder stage
# and threading it into spack.yaml's require: line.)
#
# CLANG_VERSION selects which clang/clangd/clang-format the image installs for
# IDE support. It resolves from Debian's own archive when available, and only
# falls back to apt.llvm.org otherwise.
#
# Usage:
#   make build              Build the devcontainer image
#   make build-ci           Build the lean CI image (what GHCR publishes)
#   make shell              Start a shell in the devcontainer image
#   make clean              Remove built images
#   make help               Show this help

# Include local overrides if present (not checked in, gitignored)
-include local.mk

# Container image name
IMAGE_NAME ?= shoccs-devcontainer

# Clang major version installed for clangd/clang-format
CLANG_VERSION ?= 19

# Timezone (auto-detected, override with TZ=<value>; /etc/timezone does not
# exist on macOS, so local builds fall back to UTC)
TZ ?= $(shell cat /etc/timezone 2>/dev/null || echo UTC)

# Additional docker build arguments (e.g., --no-cache)
DOCKER_BUILD_ARGS ?=

# Common docker build arguments
COMMON_BUILD_ARGS := \
	--build-arg TZ=$(TZ) \
	--build-arg CLANG_VERSION=$(CLANG_VERSION) \
	-f .devcontainer/Dockerfile \
	$(DOCKER_BUILD_ARGS)

.PHONY: build build-ci shell clean help

## Build the devcontainer image (the Dockerfile's last stage, `final`)
build:
	DOCKER_BUILDKIT=1 docker build \
		$(COMMON_BUILD_ARGS) \
		-t $(IMAGE_NAME):latest \
		.

## Build the lean CI image, mirroring .github/workflows/devcontainer-image.yml
build-ci:
	DOCKER_BUILDKIT=1 docker build \
		$(COMMON_BUILD_ARGS) \
		--target ci \
		-t $(IMAGE_NAME):ci \
		.

## Start an interactive shell in the devcontainer image
shell:
	docker run --rm -it \
		-v $(CURDIR):/workspace \
		$(IMAGE_NAME):latest

## Remove built container images
clean:
	-docker rmi $(IMAGE_NAME):latest 2>/dev/null
	-docker rmi $(IMAGE_NAME):ci 2>/dev/null

## Show available targets
help:
	@echo "SHOCCS devcontainer build targets:"
	@echo ""
	@echo "  make build          Build the devcontainer image"
	@echo "  make build-ci       Build the lean CI image (published to GHCR)"
	@echo ""
	@echo "  make shell          Shell into the devcontainer image"
	@echo ""
	@echo "  make clean          Remove built images"
	@echo ""
	@echo "Variables: IMAGE_NAME, CLANG_VERSION, TZ, DOCKER_BUILD_ARGS"
