#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
echo "Python path: $(command -v $PYTHON_CMD)"
echo ""

# Verification function: check image/folder count in images directory
verify_dataset() {
    local category=$1
    local expected_count=$2
    local count_type=$3  # "files" or "dirs" or "both"
    
    local images_dir="datasets/${category}/images"
    
    if [ ! -d "$images_dir" ]; then
        echo "  ✗ Verification failed: $images_dir does not exist"
        return 1
    fi
    
    local actual_count=0
    if [ "$count_type" = "dirs" ]; then
        # Count directories (for DocILE)
        actual_count=$(find "$images_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    elif [ "$count_type" = "both" ]; then
        # Count both files and directories (for Commercial)
        local file_count=$(find "$images_dir" -mindepth 1 -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.bmp" \) | wc -l | tr -d ' ')
        local dir_count=$(find "$images_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
        actual_count=$((file_count + dir_count))
    else
        # Count image files
        actual_count=$(find "$images_dir" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.bmp" \) | wc -l | tr -d ' ')
    fi
    
    if [ "$actual_count" -eq "$expected_count" ]; then
        echo "  ✓ Verification passed: $category - Expected: $expected_count, Actual: $actual_count"
        return 0
    else
        echo "  ✗ Verification failed: $category - Expected: $expected_count, Actual: $actual_count"
        return 1
    fi
}

echo "=========================================="
echo "Starting dataset processing"
echo "=========================================="
echo ""


# Define execution order for all processing scripts
SCRIPTS=(
    "datasets_process/process_sibr.py"
    "datasets_process/process_nanonets_kie.py"
    "datasets_process/process_hw_forms.py"
    "datasets_process/process_poie.py"
    "datasets_process/process_ephoie.py"
    "datasets_process/process_funsd.py"
    "datasets_process/process_cord.py"
    "datasets_process/process_sroie.py"
    "datasets_process/process_cell.py"
    "datasets_process/process_deepform.py"
    "datasets_process/process_docile.py"
)

# Run each script
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "----------------------------------------"
        echo "Running: $script"
        echo "----------------------------------------"
        "$PYTHON_CMD" "$script"
        echo ""
    else
        echo "Warning: File does not exist: $script"
        echo ""
    fi
done

# After all scripts complete, verify total count for each category
echo "=========================================="
echo "Verifying datasets"
echo "=========================================="
echo ""

# Define expected total count for each category (sum of all dataset contributions)
# Commercial contains files (SIBR) and directories (DocILE), use "both" type
verify_dataset "Commercial" "620" "both"
verify_dataset "Retail" "347" "files"
verify_dataset "Catering-Services" "212" "files"
verify_dataset "Accommodation" "40" "files"
verify_dataset "Administrative" "385" "files"
verify_dataset "Education" "320" "files"
verify_dataset "Postal-Label" "500" "files"
verify_dataset "Advertisement" "71" "files"
verify_dataset "Tax-Compliant" "987" "files"
verify_dataset "Medical-Services" "240" "files"
verify_dataset "Nutrition-Label" "750" "files"

echo ""
echo "=========================================="
echo "All dataset processing completed"
echo "=========================================="
