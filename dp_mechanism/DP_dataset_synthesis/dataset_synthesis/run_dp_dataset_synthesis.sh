#!/bin/bash
# run_dp_dataset_synthesis.sh - Unified script for DP dataset synthesis projects
# Usage: ./run_dp_dataset_synthesis.sh [PROJECT] [COMMAND] [ARGS...]

BASE_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
SCRIPTS_DIR="${BASE_DIR}/scripts"

# Project configurations
declare -A PROJECT_SCRIPTS=(
    ["DPMLBench"]="run_dpmlbench.sh"
    ["DPSDA"]="run_dpsda.sh" 
    ["PrivImage"]="run_privimage.sh"
)

# Show usage information
show_help() {
    echo "Usage: $0 [PROJECT] [COMMAND] [ARGS...]"
    echo "Available projects:"
    echo "  DPMLBench   - Holistic DPML evaluation framework"
    echo "  DPSDA       - Foundation model-based DP synthesis"
    echo "  PrivImage   - DP image generation with diffusion models"
    echo ""
    echo "Common commands (project-specific may vary):"
    echo "  init        Initialize environment"
    echo "  train       Run training pipeline"
    echo "  eval        Generate samples and evaluate"
    echo "  data        Prepare datasets (PrivImage only)"
    exit 1
}

# Validate project selection
validate_project() {
    local project=$1
    if [[ ! -v PROJECT_SCRIPTS[$project] ]]; then
        echo "Error: Invalid project '$project'"
        show_help
    fi
}

# Main execution
if [[ $# -lt 2 ]]; then
    show_help
fi

PROJECT=$1
COMMAND=$2
shift 2

validate_project $PROJECT

SCRIPT_PATH="${SCRIPTS_DIR}/${PROJECT_SCRIPTS[$PROJECT]}"

# Verify script existence
if [[ ! -f $SCRIPT_PATH ]]; then
    echo "Error: Script not found for $PROJECT at $SCRIPT_PATH"
    exit 1
fi

# Execute target script with remaining arguments
case $COMMAND in
    "init"|"train"|"eval"|"data")
        echo "Executing $PROJECT $COMMAND..."
        bash $SCRIPT_PATH $COMMAND "$@"
        ;;
    *)
        echo "Executing custom command for $PROJECT..."
        bash $SCRIPT_PATH $COMMAND "$@"
        ;;
esac