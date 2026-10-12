#!/usr/bin/env fish
# Helper script to run Claude Code development container
#
# Usage:
#   ./run-dev-container.fish              Run the devcontainer image
#   ./run-dev-container.fish my-image:tag Run a custom container image
#
# The gcc/clang flavours this script used to accept are gone: they mapped to
# `make build-gcc` / `make build-clang`, which produced identical images
# because the SPACK_COMPILER build arg they passed was never declared by the
# Dockerfile. The dependency tree is built with GCC, fixed by the
# `packages: all: require:` line in .devcontainer/spack.yaml.
#
# Security posture (see .devcontainer/README or docs for the rationale):
#   - No git push credentials enter the container. ~/.ssh is NOT mounted and
#     no ssh-agent socket is forwarded; commit inside the container, push from
#     the host (where the PreToolUse push guard runs).
#   - Claude Code's Bash sandbox (bubblewrap) runs inside the container. It
#     needs user + mount namespaces, which Docker's default seccomp profile
#     blocks for a process without CAP_SYS_ADMIN, so the container runs with
#     .devcontainer/seccomp-bwrap.json: Docker's default profile plus one
#     rule allowing unshare/clone/mount/umount2/pivot_root.
#   - no-new-privileges: setuid binaries (sudo) cannot raise privileges, so
#     nothing an agent starts can become root. For root, use from the host:
#         docker exec -it -u 0 <name> bash
#   - No NET_ADMIN/NET_RAW: there is no in-container iptables firewall.
#   - The forwarded port is bound to 127.0.0.1 only.

# Ensure we're in the project directory
cd (dirname (status --current-filename))

# Claude config dir mounted as the container's ~/.claude. Override with
# CLAUDE_CONTAINER_DIR (e.g. a scratch dir when trying a new image or policy).
set -q CLAUDE_CONTAINER_DIR; or set CLAUDE_CONTAINER_DIR $HOME/.claude-container

# Ensure it exists and is private to this user (it holds the container's
# Claude credential and transcripts).
mkdir -p $CLAUDE_CONTAINER_DIR
chmod 700 $CLAUDE_CONTAINER_DIR
test -f $CLAUDE_CONTAINER_DIR/.claude.json; and chmod 600 $CLAUDE_CONTAINER_DIR/.claude.json

# Default image (should match IMAGE_NAME in Makefile, tag built by `make build`)
set image_name shoccs-devcontainer

# Resolve the container image from the argument
set -q argv[1]; or set argv[1] $image_name:latest
set container $argv[1]
set name (string split -r -m1 : $container)[1]
# A registry path such as ghcr.io/x/y is not a valid container name.
set name (string replace -a / - $name)

set seccomp_profile (pwd)/.devcontainer/seccomp-bwrap.json
if not test -f $seccomp_profile
    echo "error: $seccomp_profile not found; refusing to start without it" >&2
    exit 1
end

# Colors for output
set GREEN '\033[0;32m'
set BLUE '\033[0;34m'
set YELLOW '\033[0;33m'
set NC '\033[0m' # No Color

# Warn when the image predates the sandbox tooling (bubblewrap/socat). Claude
# is configured with sandbox.failIfUnavailable=true in its settings.json, so
# it will refuse to start in such an image rather than run unsandboxed.
set sandbox_label (docker image inspect --format '{{ index .Config.Labels "org.shoccs.claude-sandbox" }}' $container 2>/dev/null)
if test "$sandbox_label" != "1"
    echo -e "$YELLOW warning: $container lacks the claude-sandbox label (rebuild with `make build`); Claude Code will refuse to start in it.$NC" >&2
end

# Find a free host port (starting at 8000) to forward to container port 8000.
# Avoids "Bind for 127.0.0.1:8000 failed: port is already allocated" when another
# process/container already holds the port. No manual editing required.
set host_port 8000
while lsof -nP -iTCP:$host_port -sTCP:LISTEN >/dev/null 2>&1
    set host_port (math $host_port + 1)
end

echo -e "$BLUE Starting $container ...$NC"
echo -e "$GREEN Forwarding 127.0.0.1:$host_port -> container port 8000$NC"
echo ""

# Run container with all necessary configuration
docker run -it --rm \
  --name $name \
  --security-opt seccomp=$seccomp_profile \
  --security-opt no-new-privileges:true \
  -p 127.0.0.1:$host_port:8000 \
  -v (pwd):/workspace \
  -v $CLAUDE_CONTAINER_DIR:/home/user/.claude \
  -v $HOME/.gitconfig:/home/user/.gitconfig:ro \
  -v claude-code-bashhistory:/commandhistory \
  -e CLAUDE_CONFIG_DIR=/home/user/.claude \
  -e POWERLEVEL9K_DISABLE_GITSTATUS=true \
  --entrypoint /bin/bash \
  $container \
  -c "exec /bin/zsh"
