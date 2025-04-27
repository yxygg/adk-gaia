#!/bin/bash

CONDA_ENV_NAME="mcp" # Replace to your conda environment name
HOST="0.0.0.0"
PORT="9012"

echo "Attempting to activate Conda environment: $CONDA_ENV_NAME"
conda activate "$CONDA_ENV_NAME"

uvicorn src.api:app --reload --host "$HOST" --port "$PORT"

echo "Uvicorn server stopped."
exit 0
