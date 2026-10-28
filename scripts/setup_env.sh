#!/usr/bin/env bash
# Create the `sharpa` conda env: Isaac Sim 5.1 + torch 2.7 cu128 (RTX 50xx / sm_120) + IsaacLab v2.3.2
set -euo pipefail
ENV=sharpa
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$HOME/workspace/third_party"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda env list | grep -q "^$ENV " || conda create -y -n $ENV python=3.11
conda activate $ENV
pip install -U pip
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

mkdir -p "$TP"
[ -d "$TP/IsaacLab" ] || git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$TP/IsaacLab"
cd "$TP/IsaacLab"
export OMNI_KIT_ACCEPT_EULA=YES
# flatdict (isaaclab dep) needs pkg_resources, which setuptools>=81 removed
pip install "setuptools<80"
pip install --no-build-isolation flatdict==4.0.1
./isaaclab.sh --install rsl_rl
# make sure the cu128 build was not replaced by a dependency
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

cd "$ROOT" && pip install -e source/rubik_lab
python -c "import torch; x=torch.randn(8,device='cuda'); print('torch', torch.__version__, torch.cuda.get_device_name(), (x*2).sum().item())"
