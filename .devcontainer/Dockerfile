# syntax=docker/dockerfile:1.4
# ^^^ Required for BuildKit cache mounts

# =============================================================================
# STAGE 1: Base system with compilers and build tools
# =============================================================================

FROM python:3.13-slim AS base-system

# TZ has a default so the CI image (built by a workflow that passes no build
# args) does not end up with ENV TZ="".
ARG TZ=UTC
ARG CLANG_VERSION=19
ENV TZ="$TZ"

# Install ca-certificates, wget, and gnupg FIRST
# (gnupg, not gnupg2 — on trixie gnupg2 is a dummy transitional package that
# only provides gpg2 symlinks, and nothing here calls gpg2 by that name.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy corporate CA bundle (empty file is fine for non-corporate environments)
COPY .devcontainer/corporate-ca.pem /usr/local/share/ca-certificates/corporate-ca.crt
RUN update-ca-certificates

# SSL environment variables
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    UV_NATIVE_TLS=true

# Provision the requested clang version, preferring Debian's own archive and
# falling back to apt.llvm.org only when Debian does not carry it.
#
# The previous version of this step wrote the apt.llvm.org source list
# unconditionally, deriving the suite name from /etc/os-release. Two things make
# that both useless and dangerous today:
#
#   - Useless: Debian trixie ships the full clang-19 set (1:19.1.7-3+b1), and
#     clang-19 is trixie's DEFAULT clang. Meanwhile the suite the old line
#     constructed — llvm-toolchain-trixie-19 — is an empty repository upstream:
#     its Release file is valid and signed, but every Packages index is
#     zero-length. So the repo contributed no packages at all.
#   - Dangerous: apt.llvm.org has no per-codename directory for unreleased
#     Debian (forky/sid are served from /unstable/ with no codename infix, and
#     https://apt.llvm.org/forky/ is a 404). The moment the python base image
#     moves to the next Debian, the derived line would point at a nonexistent
#     suite and the following `apt-get update` would hard-fail the build —
#     precisely the breakage the codename derivation was meant to prevent.
#
# So: probe apt for a real candidate first, and only reach for apt.llvm.org (and
# only after confirming the suite exists) when Debian comes up empty. Failing
# loudly beats silently producing an image without the requested clang.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN . /etc/os-release && \
    apt-get update && \
    CANDIDATE="$(apt-cache policy "clang-${CLANG_VERSION}" 2>/dev/null | awk '/Candidate:/{print $2}')" && \
    if [ -n "$CANDIDATE" ] && [ "$CANDIDATE" != "(none)" ]; then \
        echo "clang-${CLANG_VERSION} available from Debian ${VERSION_CODENAME} ($CANDIDATE); not adding apt.llvm.org"; \
    elif wget -q --spider "https://apt.llvm.org/${VERSION_CODENAME}/dists/llvm-toolchain-${VERSION_CODENAME}-${CLANG_VERSION}/InRelease"; then \
        echo "clang-${CLANG_VERSION} not in Debian ${VERSION_CODENAME}; adding apt.llvm.org"; \
        wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | \
        gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg && \
        echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME}-${CLANG_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list; \
    else \
        echo "ERROR: clang-${CLANG_VERSION} is in neither Debian ${VERSION_CODENAME} nor apt.llvm.org's llvm-toolchain-${VERSION_CODENAME}-${CLANG_VERSION} suite" >&2; \
        exit 1; \
    fi && \
    rm -rf /var/lib/apt/lists/*

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

# Create 'user' user early (needed for spack stage).
# Login shell is bash here because zsh is only installed in the `final` stage —
# pointing uid 1000 at /bin/zsh in base-system left the published CI image with
# a login shell that does not exist. `final` re-points it to zsh once zsh is
# actually present.
ARG USERNAME=user
RUN useradd -m -u 1000 -s /bin/bash $USERNAME


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
# environment manifest pins to a tagged release containing Kokkos 5.1.1).
# Spack 1.2 accepts package repos declaring API v1.0 through v2.5, which covers
# both spack_repo/shoccs (v2.0) and spack-packages v2026.06.0 (v2.2).
ARG SPACK_VERSION=v1.2.2
RUN git clone --depth=1 -b $SPACK_VERSION https://github.com/spack/spack.git /home/$USERNAME/spack

# Copy the project-owned custom spack repo to the ABSOLUTE path referenced by
# the environment manifest's `repos:` entry.
COPY --chown=$USERNAME:$USERNAME spack_repo /home/$USERNAME/spack_repo

# Copy the spack environment file
COPY --chown=$USERNAME:$USERNAME .devcontainer/spack.yaml /home/$USERNAME/shoccs-env/spack.yaml

# Configure Spack: find system compilers (Spack 1.0 records these as externals
# in packages.yaml)
RUN /home/$USERNAME/spack/bin/spack compiler find

# Concretize and build all dependencies.
# `spack repo update builtin` forces the on-disk clone of the community package
# repo to the ref pinned under `repos: builtin:` in spack.yaml. Historically
# this was REQUIRED: Spack auto-cloned its default builtin repo on first use and
# never re-synced it to the pinned ref, so the manifest pin was silently
# ignored. Keep the call as a cheap, idempotent assertion that the checkout
# really is at the pinned ref.
# The tag is read back OUT of the manifest rather than restated here, so
# .devcontainer/spack.yaml stays the single source of truth — the previous
# arrangement duplicated a commit SHA across the two files, where only this one
# took effect and a manifest-only edit produced a silently wrong dep tree.
# That grep assumes the manifest holds exactly ONE `tag:` line (it takes the
# first match, and under `set -o pipefail` a second match risks SIGPIPE-ing sed
# when head exits early). The devcontainer-image workflow asserts that count
# before it starts a build, so the assumption fails loudly in CI, not here.
# Cache mounts persist across builds for faster rebuilds:
#   - var/spack/cache: downloaded source tarballs
#   - .spack/cache: misc caches including fetch cache
#   - .spack/bootstrap: spack's own bootstrapped tools (clingo, etc.)
#   - /tmp/user/spack-stage: build staging area
# Only config:connect_timeout is a real Spack config key — config:read_timeout
# and config:retry were accepted silently but appear nowhere in Spack's config
# schema and are read by nothing, so they have been dropped.
# NOTE: the cache-mount targets below hardcode /home/user because BuildKit does
# not expand build args in --mount targets; they must be kept in step with
# USERNAME (which is why overriding USERNAME is not supported).
RUN --mount=type=cache,target=/home/user/spack/var/spack/cache,uid=1000,gid=1000 \
    --mount=type=cache,target=/home/user/.spack/cache,uid=1000,gid=1000 \
    --mount=type=cache,target=/home/user/.spack/bootstrap,uid=1000,gid=1000 \
    --mount=type=cache,target=/tmp/user/spack-stage,uid=1000,gid=1000 \
    source /home/$USERNAME/spack/share/spack/setup-env.sh && \
    spack config add config:connect_timeout:60 && \
    if [ -n "$SPACK_MIRROR_URL" ]; then spack mirror add custom-mirror "$SPACK_MIRROR_URL"; fi && \
    spack env create shoccs-dev /home/$USERNAME/shoccs-env/spack.yaml && \
    spack env activate shoccs-dev && \
    BUILTIN_TAG="$(sed -n 's/^[[:space:]]*tag:[[:space:]]*//p' /home/$USERNAME/shoccs-env/spack.yaml | head -1)" && \
    test -n "$BUILTIN_TAG" && \
    echo "pinning builtin package repo to $BUILTIN_TAG (from spack.yaml)" && \
    spack repo update builtin --tag "$BUILTIN_TAG" && \
    spack repo list && \
    spack concretize && \
    spack install -j $(nproc) --only dependencies --fail-fast


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
#
# DO NOT add a `USER` instruction here, however tempting the symmetry with
# spack-builder and final is. ci.yml runs this image as a GitHub Actions job
# container, and Actions mounts the workspace at /__w owned by the runner uid
# (1001); as a non-root user actions/checkout fails with EACCES before any of
# the build steps run.

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

ARG TZ=UTC
ARG USERNAME=user
ARG CLAUDE_CODE_VERSION=latest
# 0.19.2, not 0.19.0/0.19.1: those two shipped broken Debian packages (0.19.0
# published only the musl .deb; 0.19.1's glibc build required 2.39). 0.19.2 was
# released specifically to rebuild them for older glibc.
ARG GIT_DELTA_VERSION=0.19.2

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
    bind9-dnsutils \
    aggregate \
    jq \
    neovim \
    openssh-client \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN pip install --no-cache-dir uv

# Configure user with sudo access. The NOPASSWD drop-in is required, not a
# convenience: useradd sets no password, so the account's password field is
# locked and a plain `sudo` would prompt for a password that can never be
# supplied. zsh is available in this stage, so make it the login shell here
# (base-system leaves it as bash — see the useradd comment there).
RUN usermod -aG sudo $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chsh -s /bin/zsh $USERNAME

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
