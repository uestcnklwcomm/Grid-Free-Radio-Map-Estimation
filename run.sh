#!/bin/bash
# Grid-Free Radio Map Estimation Shell Runner

# Switch to the script's directory (ensure relative paths are correct)
cd "$(dirname "$0")"

# Show usage if no argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [FSD | RMSeer]"
    echo "Example: $0 FSD"
    exit 1
fi

# Activate virtual environment (if available)
# source venv/bin/activate

# Run the selected script based on the argument
case "$1" in
    FSD)
        echo "Running Grid_Free_RME_FSD.py ..."
        python scripts/Grid_Free_RME_FSD.py
        ;;
    RMSeer)
        echo "Running Grid_Free_RME_RMSeer.py ..."
        python scripts/Grid_Free_RME_RMSeer.py
        ;;
    *)
        echo "Invalid argument: $1"
        echo "Options: FSD | RMSeer"
        exit 1
        ;;
esac
