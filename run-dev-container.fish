#!/usr/bin/env fish
# Helper script to run Claude Code development container
#
# Usage:
#   ./run-dev-container.fish              Run the default (gcc) container
#   ./run-dev-container.fish gcc          Run the gcc container
#   ./run-dev-container.fish clang        Run the clang container
#   ./run-dev-container.fish my-image:tag Run a custom container image

# Ensure we're in the project directory
cd (dirname (status --current-filename))

# Ensure ~/.claude-container exists
mkdir -p $HOME/.claude-container

# Default image name (should match IMAGE_NAME in Makefile)
set image_name shoccs-devcontainer

# Resolve the container image from the argument
set -q argv[1]; or set argv[1] gcc
switch $argv[1]
    case gcc clang
        set container $image_name:$argv[1]
    case '*'
        set container $argv[1]
end
set name (string split -r -m1 : $container)[1]

# Colors for output
set GREEN '\033[0;32m'
set BLUE '\033[0;34m'
set NC '\033[0m' # No Color

echo -e "$BLUE Starting $container ...$NC"
echo ""

# Run container with all necessary configuration
docker run -it --rm \
  --name $name \
  --cap-add=NET_ADMIN \
  --cap-add=NET_RAW \
  -p 8000:8000 \
  -v (pwd):/workspace \
  -v $HOME/.claude-container:/home/user/.claude \
  -v $HOME/.gitconfig:/home/user/.gitconfig:ro \
  -v $HOME/.ssh:/home/user/.ssh:ro \
  -v claude-code-bashhistory:/commandhistory \
  -e CLAUDE_CONFIG_DIR=/home/user/.claude \
  -e POWERLEVEL9K_DISABLE_GITSTATUS=true \
  --entrypoint /bin/bash \
  $container \
  -c "exec /bin/zsh"
