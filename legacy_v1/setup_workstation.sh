#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Capstone Workstation Setup Script
# Run this every session after SSH-ing in:
#   bash setup_workstation.sh
# ─────────────────────────────────────────────────────────────

set -e  # stop on any error

echo "=== Step 1: Creating working directory ==="
mkdir -p ~/capstone
cd ~/capstone

echo "=== Step 2: Setting up Python 3.8 venv ==="
python3.8 -m venv env
source env/bin/activate

echo "=== Step 3: Installing pip for Python 3.8 ==="
curl https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py
python get-pip.py
rm get-pip.py

echo "=== Step 4: Installing numpy and spacy first (order matters) ==="
pip install numpy==1.24.4
pip install spacy==3.2.1

echo "=== Step 5: Installing opennyai ==="
pip install opennyai

echo "=== Step 6: Installing remaining requirements ==="
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --ignore-installed torch spacy opennyai numpy

echo "=== Step 7: Force pinning pydantic (critical for spacy) ==="
pip install pydantic==1.7.4 --force-reinstall

echo "=== Step 8: Pulling latest code from GitHub ==="
git config --global user.email "smruthibs1310@gmail.com"
git config --global user.name "smruthi-bs"
git init 2>/dev/null || true
git remote add origin https://github.com/smruthi-bs/capstone 2>/dev/null || true
git fetch
git checkout main 2>/dev/null || git pull origin main

echo "=== Step 9: Downloading model files from HuggingFace ==="
hf auth login
hf download Capstone-LawAI/capstone-legal --local-dir . --repo-type model
hf download Capstone-LawAI/legal-data --local-dir . --repo-type dataset

echo "=== Step 10: Creating OpenNyAI temp directory (critical fix) ==="
mkdir -p /home/mluser/.opennyai/temp_hsln/pubmed-20k/

echo ""
echo "✅ Setup complete! Now run:"
echo "   source ~/capstone/env/bin/activate"
echo "   cd ~/capstone"
echo "   python opennyai_ablation.py"
