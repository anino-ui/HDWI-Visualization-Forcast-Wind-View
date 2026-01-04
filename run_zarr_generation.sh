#!/bin/bash

# =============================================================================
# Automated Zarr Generation Script with Execution Time Logging
# This script runs the zarr generation process and logs execution time
# =============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/update_tiles_and_zarr.py"
LOG_FILE="$SCRIPT_DIR/zarr_generation_log.txt"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Start timing
START_TIME=$(date +%s)
START_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "========================================" >> "$LOG_FILE"
echo "Zarr Generation Started: $START_TIMESTAMP" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Run the Python script and capture output
python3 "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1
PYTHON_EXIT_CODE=$?

# End timing
END_TIME=$(date +%s)
END_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((END_TIME - START_TIME))

# Calculate hours, minutes, seconds
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Log results
echo "" >> "$LOG_FILE"
echo "Zarr Generation Completed: $END_TIMESTAMP" >> "$LOG_FILE"
echo "Exit Code: $PYTHON_EXIT_CODE" >> "$LOG_FILE"
echo "Execution Time: ${HOURS}h ${MINUTES}m ${SECONDS}s (Total: ${DURATION} seconds)" >> "$LOG_FILE"

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS" >> "$LOG_FILE"
else
    echo "Status: FAILED" >> "$LOG_FILE"
fi

echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Exit with the same code as the Python script
exit $PYTHON_EXIT_CODE
