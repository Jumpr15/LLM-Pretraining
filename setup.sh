#!/bin/bash
set -e

# Installs Tmux for detachable sessions that persist without VM connection
sudo apt update
sudo apt install -y tmux

# Installs dependencies
uv sync
# Installs Liger Kernel (Only works with CUDA devices)
uv add liger-kernel
