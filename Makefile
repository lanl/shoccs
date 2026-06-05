# Makefile for building SHOCCS devcontainer images.
#
# Wraps Docker build commands to simplify building containers with
# different compiler toolchains. Both GCC and Clang are always installed
# in the container; the SPACK_COMPILER arg controls which compiler spack
# uses to build the dependency tree. CLANG_VERSION controls which clang
# release is installed from the LLVM apt repository (apt.llvm.org).
#
# Usage:
#   make build              Build gcc image (default)
#   make build-gcc          Build gcc image
#   make build-clang        Build clang image
#   make build-all          Build both gcc and clang images
#   make shell              Start a shell in the default container
#   make shell-gcc          Start a shell in the gcc container
#   make shell-clang        Start a shell in the clang container
#   make clean              Remove built images
#   make help               Show this help

# Include local overrides if present (not checked in, gitignored)
-include local.mk

# Container image name
IMAGE_NAME ?= shoccs-devcontainer

# Default compiler (gcc or clang)
COMPILER ?= gcc

# Clang major version to install from apt.llvm.org
CLANG_VERSION ?= 19

# Timezone (auto-detected, override with TZ=<value>)
TZ ?= $(shell cat /etc/timezone 2>/dev/null || echo UTC)

# Additional docker build arguments (e.g., --no-cache)
DOCKER_BUILD_ARGS ?=

# Compiler-specific spack specs
GCC_SPACK_SPEC   ?= gcc@14.2.0
CLANG_SPACK_SPEC ?= clang@$(CLANG_VERSION)

# Common docker build arguments
COMMON_BUILD_ARGS := \
	--build-arg TZ=$(TZ) \
	--build-arg CLANG_VERSION=$(CLANG_VERSION) \
	-f .devcontainer/Dockerfile \
	$(DOCKER_BUILD_ARGS)

.PHONY: build build-gcc build-clang build-all \
        shell shell-gcc shell-clang \
        clean help

## Build the container image for the selected compiler (default: gcc)
build: build-$(COMPILER)

## Build the gcc-based container image
build-gcc:
	DOCKER_BUILDKIT=1 docker build \
		$(COMMON_BUILD_ARGS) \
		--build-arg SPACK_COMPILER=$(GCC_SPACK_SPEC) \
		-t $(IMAGE_NAME):gcc \
		.

## Build the clang-based container image
build-clang:
	DOCKER_BUILDKIT=1 docker build \
		$(COMMON_BUILD_ARGS) \
		--build-arg SPACK_COMPILER=$(CLANG_SPACK_SPEC) \
		-t $(IMAGE_NAME):clang \
		.

## Build both gcc and clang container images
build-all: build-gcc build-clang

## Start an interactive shell in the selected compiler's container (default: gcc)
shell: shell-$(COMPILER)

## Start an interactive shell in the gcc container
shell-gcc:
	docker run --rm -it \
		-v $(CURDIR):/workspace \
		$(IMAGE_NAME):gcc

## Start an interactive shell in the clang container
shell-clang:
	docker run --rm -it \
		-v $(CURDIR):/workspace \
		$(IMAGE_NAME):clang

## Remove built container images
clean:
	-docker rmi $(IMAGE_NAME):gcc 2>/dev/null
	-docker rmi $(IMAGE_NAME):clang 2>/dev/null

## Show available targets
help:
	@echo "SHOCCS devcontainer build targets:"
	@echo ""
	@echo "  make build          Build container (default: gcc)"
	@echo "  make build-gcc      Build gcc container"
	@echo "  make build-clang    Build clang container"
	@echo "  make build-all      Build both gcc and clang containers"
	@echo ""
	@echo "  make shell          Shell into container (default: gcc)"
	@echo "  make shell-gcc      Shell into gcc container"
	@echo "  make shell-clang    Shell into clang container"
	@echo ""
	@echo "  make clean          Remove built images"
	@echo ""
	@echo "Variables (override via command line, environment, or local.mk):"
	@echo "  COMPILER=gcc|clang       Select default compiler (default: gcc)"
	@echo "  CLANG_VERSION=<major>    Clang version from apt.llvm.org (default: 19)"
	@echo "  IMAGE_NAME=<name>        Container image name (default: shoccs-devcontainer)"
	@echo "  GCC_SPACK_SPEC=<spec>    GCC spack compiler spec (default: gcc@14.2.0)"
	@echo "  CLANG_SPACK_SPEC=<spec>  Clang spack compiler spec (default: clang@CLANG_VERSION)"
	@echo "  TZ=<timezone>            Timezone (default: auto-detected)"
	@echo "  DOCKER_BUILD_ARGS=       Additional docker build arguments"
