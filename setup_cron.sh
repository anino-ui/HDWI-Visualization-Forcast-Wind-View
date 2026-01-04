#!/bin/bash

# =============================================================================
# Cron Job Setup Script for Zarr Generation
# This script helps set up the daily cron job at 2:40 AM
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZARR_SCRIPT="$SCRIPT_DIR/run_zarr_generation.sh"

# Make the zarr generation script executable
chmod +x "$ZARR_SCRIPT"

# Cron entry to run at 2:40 AM daily
CRON_ENTRY="40 2 * * * $ZARR_SCRIPT"

echo "========================================="
echo "Cron Job Setup for Zarr Generation"
echo "========================================="
echo ""
echo "This will add a cron job to run zarr generation at 2:40 AM daily."
echo ""
echo "Cron entry:"
echo "$CRON_ENTRY"
echo ""

# Check if cron entry already exists
if crontab -l 2>/dev/null | grep -q "$ZARR_SCRIPT"; then
    echo "⚠️  Cron job already exists!"
    echo ""
    echo "Current crontab entries for this script:"
    crontab -l 2>/dev/null | grep "$ZARR_SCRIPT"
    echo ""
    read -p "Do you want to update it? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping cron job setup."
        exit 0
    fi
    # Remove old entry
    crontab -l 2>/dev/null | grep -v "$ZARR_SCRIPT" | crontab -
fi

# Add new cron entry
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo ""
echo "✅ Cron job added successfully!"
echo ""
echo "Current crontab:"
crontab -l
echo ""
echo "The script will run at 2:40 AM every day and log to:"
echo "$SCRIPT_DIR/zarr_generation_log.txt"
echo ""
echo "To view logs:"
echo "  tail -f $SCRIPT_DIR/zarr_generation_log.txt"
echo ""
echo "To remove this cron job:"
echo "  crontab -e"
echo "  (then delete the line with $ZARR_SCRIPT)"
echo ""
