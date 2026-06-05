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
#   - Spack version changes
#
# All SHOCCS dependencies are in the standard spack builtin repo.
# No custom spack_repo is needed.

FROM base-system AS spack-builder

ARG USERNAME=user
ARG SPACK_MIRROR_URL=""
USER $USERNAME
WORKDIR /home/$USERNAME

# Clone and configure Spack (use develop branch for Kokkos 5.0+ recipes)
RUN git clone --depth=1 -b develop https://github.com/spack/spack.git /home/$USERNAME/spack

# Copy the spack environment file
COPY --chown=$USERNAME:$USERNAME .devcontainer/spack.yaml /home/$USERNAME/shoccs-env/spack.yaml

# Configure Spack: find system compilers
RUN /home/$USERNAME/spack/bin/spack compiler find

# Concretize and build all dependencies
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
    spack concretize && \
    spack install -j $(nproc)


# =============================================================================
# STAGE 3: Final devcontainer image
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
    UV_NATIVE_TLS=true

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
    echo 'spack env activate shoccs-dev' >> /home/$USERNAME/.zshrc

WORKDIR /workspace
CMD ["/bin/zsh"]
