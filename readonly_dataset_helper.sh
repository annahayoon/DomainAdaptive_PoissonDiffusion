#!/bin/bash

# Read-only dataset helper script
# This script provides safe access to the dataset without risk of deletion

DATASET_PATH="/home/jilab/Jae/dataset"
ORIGINAL_PATH="/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/processed"

echo "=== Read-Only Dataset Helper ==="
echo "Dataset location: $DATASET_PATH"
echo "Original location: $ORIGINAL_PATH"
echo ""

# Function to safely list files
list_files() {
    echo "Listing files in dataset:"
    ls -la "$DATASET_PATH"
}

# Function to safely copy files (read-only operation)
copy_files() {
    if [ -z "$1" ]; then
        echo "Usage: $0 copy <destination>"
        echo "Example: $0 copy /home/jilab/Jae/temp_training_data"
        exit 1
    fi
    echo "Copying files to: $1"
    cp -r "$DATASET_PATH"/* "$1"
    echo "Files copied successfully!"
}

# Function to show dataset info
show_info() {
    echo "Dataset Information:"
    echo "Total size: $(du -sh "$DATASET_PATH" 2>/dev/null || echo "Unable to calculate")"
    echo "File count: $(find "$DATASET_PATH" -type f | wc -l)"
    echo "Directory structure:"
    tree "$DATASET_PATH" 2>/dev/null || ls -R "$DATASET_PATH"
}

# Function to create a safe working copy
create_working_copy() {
    if [ -z "$1" ]; then
        echo "Usage: $0 working_copy <destination>"
        echo "Example: $0 working_copy /home/jilab/Jae/temp_working_data"
        exit 1
    fi
    echo "Creating working copy at: $1"
    mkdir -p "$1"
    cp -r "$DATASET_PATH"/* "$1"
    echo "Working copy created! You can safely modify files in: $1"
}

case "$1" in
    "list")
        list_files
        ;;
    "copy")
        copy_files "$2"
        ;;
    "info")
        show_info
        ;;
    "working_copy")
        create_working_copy "$2"
        ;;
    *)
        echo "Usage: $0 {list|copy|info|working_copy}"
        echo ""
        echo "Commands:"
        echo "  list          - List files in the dataset"
        echo "  copy <dest>   - Copy files to destination"
        echo "  info          - Show dataset information"
        echo "  working_copy <dest> - Create a safe working copy"
        echo ""
        echo "Examples:"
        echo "  $0 list"
        echo "  $0 copy /home/jilab/Jae/temp_data"
        echo "  $0 info"
        echo "  $0 working_copy /home/jilab/Jae/my_training_data"
        ;;
esac
