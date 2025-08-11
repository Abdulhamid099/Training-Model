#!/usr/bin/env bash
set -euo pipefail

# Clone the Cashew Hugging Face Space locally
if [[ ! -d Cashew ]]; then
  git clone https://huggingface.co/spaces/Abdulhamid75/Cashew
else
  echo "Directory 'Cashew' already exists. Pulling latest..."
  (cd Cashew && git pull --ff-only)
fi