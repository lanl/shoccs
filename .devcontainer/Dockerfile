# syntax=docker/dockerfile:1.4
# ^^^ Required for BuildKit cache mounts

# =============================================================================
# STAGE 1: Base system with compilers and build tools
# =============================================================================

FROM python:3.11-slim AS base-system

ARG TZ
ARG CLANG_VERSION=19
ENV TZ="$TZ"

# Install ca-certificates, wget, and gnupg2 FIRST
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    gnupg2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy corporate CA bundle (empty file is fine for non-corporate environments)
COPY .devcontainer/corporate-ca.pem /usr/local/share/ca-certificates/corporate-ca.crt
RUN update-ca-certificates

# SSL environment variables
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    UV_NATIVE_TLS=true

# Add the LLVM apt repository for the requested clang version.
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | \
    gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-${CLANG_VERSION} main" \
    > /etc/apt/sources.list.d/llvm.list

# Install compilers and build tools
# GCC provides gfortran and serves as the spack bootstrap compiler.
# Clang provides clangd and clang-format for IDE support.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    gcc \
    g++ \
    gfortran \
    clang-${CLANG_VERSION} \
    clangd-${CLANG_VERSION} \
    clang-format-${CLANG_VERSION} \
    lld-${CLANG_VERSION} \
    libc++-${CLANG_VERSION}-dev \
    libc++abi-${CLANG_VERSION}-dev \
    make \
    cmake \
    ninja-build \
    bzip2 \
    patch \
    xz-utils \
    unzip \
    locales \
    file \
    swig \
    && ln -sf /usr/bin/clang-${CLANG_VERSION} /usr/local/bin/clang \
    && ln -sf /usr/bin/clang++-${CLANG_VERSION} /usr/local/bin/clang++ \
    && ln -sf /usr/bin/clangd-${CLANG_VERSION} /usr/local/bin/clangd \
    && ln -sf /usr/bin/clang-format-${CLANG_VERSION} /usr/local/bin/clang-format \
    && ln -sf /usr/bin/lld-${CLANG_VERSION} /usr/local/bin/lld \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Create 'user' user early (needed for spack stage)
ARG USERNAME=user
RUN useradd -m -u 1000 -s /bin/zsh $USERNAME || true


# =============================================================================
# STAGE 2: Spack dependency builder
# =============================================================================
# This stage only rebuilds when:
#   - base-system changes (compilers, etc.)
#   - spack.yaml changes (dependency specifications)
#   - the spack_repo/ recipe changes
#   - the pinned Spack / spack-packages versions change
#
# SHOCCS dependencies are provisioned from a PROJECT-OWNED custom spack repo
# (spack_repo/shoccs, checked into the git repo) that declares a single
# `shoccs` recipe. The community spack-packages repo is pinned (in spack.yaml)
# to a develop commit that ships Kokkos 5.1+ (the Graph API the solver needs;
# no tagged spack-packages release carries 5.1.x yet).

FROM base-system AS spack-builder

ARG USERNAME=user
ARG SPACK_MIRROR_URL=""
USER $USERNAME
WORKDIR /home/$USERNAME

# Clone Spack at a pinned 1.x release for reproducibility (since Spack 1.0 the
# builtin package recipes live in the separate spack-packages repo, which the
# environment manifest pins to a develop commit containing Kokkos 5.1).
RUN git clone --depth=1 -b v1.1.1 https://github.com/spack/spack.git /home/$USERNAME/spack

# Copy the project-owned custom spack repo to the ABSOLUTE path referenced by
# the environment manifest's `repos:` entry.
COPY --chown=$USERNAME:$USERNAME spack_repo /home/$USERNAME/spack_repo

# Copy the spack environment file
COPY --chown=$USERNAME:$USERNAME .devcontainer/spack.yaml /home/$USERNAME/shoccs-env/spack.yaml

# Configure Spack: find system compilers (Spack 1.0 records these as externals
# in packages.yaml)
RUN /home/$USERNAME/spack/bin/spack compiler find

# Concretize and build all dependencies.
# NOTE: `spack repo update builtin` is REQUIRED. Spack auto-clones the DEFAULT
# builtin repo (this Spack release pins releases/v2025.11, which has no kokkos
# 5.1) on first use and does NOT re-sync that on-disk clone to the commit pinned
# under `repos: builtin:` in spack.yaml. `spack repo update` forces the checkout
# to the pinned commit, which ships kokkos 5.1.1.
# The commit SHA on the `spack repo update builtin --commit` line below is what
# actually takes effect; keep it in sync with the `builtin::` commit in
# .devcontainer/spack.yaml (editing only the manifest is silently ignored).
# Cache mounts persist across builds for faster rebuilds:
#   - var/spack/cache: downloaded source tarballs
#   - .spack/cache: misc caches including fetch cache
#   - .spack/bootstrap: spack's own bootstrapped tools (clingo, etc.)
#   - /tmp/user/spack-stage: build staging area
SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/home/user/spack/var/spack/cache,uid=1000,gid=1000 \
    --mount=type=cache,target=/home/user/.spack/cache,uid=1000,gid=1000 \
    --mount=type=cache,target=/home/user/.spack/bootstrap,uid=1000,gid=1000 \
    --mount=type=cache,target=/tmp/user/spack-stage,uid=1000,gid=1000 \
    source /home/$USERNAME/spack/share/spack/setup-env.sh && \
    spack config add config:connect_timeout:60 && \
    spack config add config:read_timeout:300 && \
    spack config add config:retry:3 && \
    if [ -n "$SPACK_MIRROR_URL" ]; then spack mirror add custom-mirror "$SPACK_MIRROR_URL"; fi && \
    spack env create shoccs-dev /home/$USERNAME/shoccs-env/spack.yaml && \
    spack env activate shoccs-dev && \
    spack repo update builtin --commit 1b13532cdcb78de32ef23a83a05d2ab144bd0702 && \
    spack repo list && \
    spack concretize && \
    spack install -j $(nproc) --only dependencies


# =============================================================================
# STAGE 3: Lean CI image
# =============================================================================
# Published to GHCR by .github/workflows/devcontainer-image.yml and consumed
# by the cpp job in .github/workflows/ci.yml. Toolchain (base-system) plus the
# spack-built dependency tree only — no Claude Code, shell customizations, or
# other devcontainer conveniences, so the image stays small and contains
# nothing unredistributable.
#
# This stage must come BEFORE `final`: devcontainer.json and the Makefile
# build without --target, which selects the last stage in the file.

FROM base-system AS ci

ARG USERNAME=user

# ccache backs the actions/cache-persisted compiler cache in CI
RUN apt-get update && apt-get install -y --no-install-recommends \
    ccache \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# The spack tree must stay at its build-time path: the environment view and
# the library rpaths point into /home/user/spack/opt. The bootstrap store
# (~/.spack) is not copied — CI never runs spack itself; cmake resolves all
# dependencies through the view.
COPY --from=spack-builder --chown=$USERNAME:$USERNAME /home/$USERNAME/spack /home/$USERNAME/spack
COPY --from=spack-builder --chown=$USERNAME:$USERNAME /home/$USERNAME/shoccs-env /home/$USERNAME/shoccs-env
COPY --from=spack-builder --chown=$USERNAME:$USERNAME /home/$USERNAME/spack_repo /home/$USERNAME/spack_repo

ENV CMAKE_PREFIX_PATH=/home/$USERNAME/spack/var/spack/environments/shoccs-dev/.spack-env/view

# =============================================================================
# STAGE 4: Final devcontainer image
# =============================================================================

FROM base-system AS final

ARG TZ
ARG USERNAME=user
ARG CLAUDE_CODE_VERSION=latest
ARG GIT_DELTA_VERSION=0.18.2

# Install developer convenience tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    less \
    procps \
    sudo \
    fzf \
    zsh \
    man-db \
    gh \
    iptables \
    ipset \
    iproute2 \
    dnsutils \
    aggregate \
    jq \
    neovim \
    openssh-client \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN pip install --no-cache-dir uv

# Configure user with sudo access
RUN usermod -aG sudo $USERNAME

# Persist bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory

# Create workspace and config directories
RUN mkdir -p /workspace /home/$USERNAME/.claude \
    && chown -R $USERNAME:$USERNAME /workspace /home/$USERNAME/.claude

# Install git-delta
RUN ARCH=$(dpkg --print-architecture) && \
    wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# =============================================================================
# Copy artifacts from spack-builder
# =============================================================================

COPY --from=spack-builder --chown=$USERNAME:$USERNAME /home/$USERNAME/spack /home/$USERNAME/spack
COPY --from=spack-builder --chown=$USERNAME:$USERNAME /home/$USERNAME/shoccs-env /home/$USERNAME/shoccs-env
COPY --from=spack-builder --chown=$USERNAME:$USERNAME /home/$USERNAME/.spack /home/$USERNAME/.spack
# The custom repo must remain at the absolute path the environment's repos:
# entry points at, so `spack env activate shoccs-dev` can load the recipe.
COPY --from=spack-builder --chown=$USERNAME:$USERNAME /home/$USERNAME/spack_repo /home/$USERNAME/spack_repo

# =============================================================================
# User configuration
# =============================================================================

USER $USERNAME

# Install Claude Code CLI via native installer
RUN curl -fsSL https://claude.ai/install.sh | bash -s ${CLAUDE_CODE_VERSION}

# Environment variables
ENV DEVCONTAINER=true \
    ENABLE_LSP_TOOLS=1 \
    PATH=$PATH:/home/$USERNAME/.local/bin \
    SHELL=/bin/zsh \
    EDITOR=nvim \
    VISUAL=nvim \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    UV_NATIVE_TLS=true \
    CMAKE_PREFIX_PATH=/home/$USERNAME/spack/var/spack/environments/shoccs-dev/.spack-env/view

# Install Oh My Zsh
RUN git clone --depth=1 https://github.com/ohmyzsh/ohmyzsh.git /home/$USERNAME/.oh-my-zsh

# Create .zshrc with all configurations
RUN echo 'export ZSH="/home/user/.oh-my-zsh"' > /home/$USERNAME/.zshrc && \
    echo 'ZSH_THEME="robbyrussell"' >> /home/$USERNAME/.zshrc && \
    echo 'plugins=(git)' >> /home/$USERNAME/.zshrc && \
    echo 'source $ZSH/oh-my-zsh.sh' >> /home/$USERNAME/.zshrc && \
    echo '' >> /home/$USERNAME/.zshrc && \
    echo '# Terminal and locale configuration' >> /home/$USERNAME/.zshrc && \
    echo '[ -z "$TERM" ] && export TERM=xterm' >> /home/$USERNAME/.zshrc && \
    echo 'export LANG="en_US.UTF-8"' >> /home/$USERNAME/.zshrc && \
    echo 'export LANGUAGE="en_US:en"' >> /home/$USERNAME/.zshrc && \
    echo 'export LC_ALL="en_US.UTF-8"' >> /home/$USERNAME/.zshrc && \
    echo '' >> /home/$USERNAME/.zshrc && \
    echo '# Command history persistence' >> /home/$USERNAME/.zshrc && \
    echo 'export HISTFILE=/commandhistory/.bash_history' >> /home/$USERNAME/.zshrc && \
    echo '' >> /home/$USERNAME/.zshrc && \
    echo '# Spack initialization' >> /home/$USERNAME/.zshrc && \
    echo 'source /home/user/spack/share/spack/setup-env.sh' >> /home/$USERNAME/.zshrc && \
    echo 'spack env activate shoccs-dev || echo "warning: could not activate shoccs-dev; run: spack env list"' >> /home/$USERNAME/.zshrc

WORKDIR /workspace
CMD ["/bin/zsh"]
