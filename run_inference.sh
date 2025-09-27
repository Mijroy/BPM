#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Set your DICOM image paths here - MODIFY THESE PATHS
RCC_IMAGE="/mnt/BPM/alldata/phase3_ge_origin/darwin/2d_proc_batch_1/origin/2d_proc.00034.rcc.dcm"
LCC_IMAGE="/mnt/BPM/alldata/phase3_ge_origin/darwin/2d_proc_batch_1/origin/2d_proc.00034.lcc.dcm"
RMLO_IMAGE="/mnt/BPM/alldata/phase3_ge_origin/darwin/2d_proc_batch_1/origin/2d_proc.00034.rmlo.dcm"
LMLO_IMAGE="/mnt/BPM/alldata/phase3_ge_origin/darwin/2d_proc_batch_1/origin/2d_proc.00034.lmlo.dcm"

# Segmentation model path
SEG_MODEL_PATH="/workspace/bpm_classification/criteria/model_5-6/best_checkpoint_final.pt"

# Output directory for results
OUTPUT_DIR="/workspace/bpm_classification/result/"
mkdir -p "$OUTPUT_DIR"

# Create log file
IDENTIFIER="$(basename "$RCC_IMAGE" | awk -F. '{print $2}')"
LOG_FILE="${OUTPUT_DIR}/inference_log_${IDENTIFIER}.txt"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start logging
{
    echo "=========================================="
    echo "BPM Classification Integrated Inference"
    echo "=========================================="
    echo "Start Time: $(date)"
    echo ""
    echo "Configuration:"
    echo "  - RCC Image: $RCC_IMAGE"
    echo "  - LCC Image: $LCC_IMAGE"
    echo "  - RMLO Image: $RMLO_IMAGE"
    echo "  - LMLO Image: $LMLO_IMAGE"
    echo "  - Seg Model: $SEG_MODEL_PATH"
    echo "  - Output Dir: $OUTPUT_DIR"
    echo ""
    echo "Requirements:"
    echo "  - Time: < 5 seconds for inference only"
    echo "  - GPU Memory: < 16 GB"
    echo "  - RAM: < 128 GB"
    echo ""
    echo "Test Setup:"
    echo "  - Models 1-4: 4 classification models"
    echo "  - Models 5-6: 2 segmentation models (Model 5 + Model 6)"
    echo "  - 4 images (RCC, LCC, RMLO, LMLO)"
    echo "  - Expected operations: 16 total"
    echo "    * Models 1-4: 12 classifications"
    echo "    * Model 5: 2 inferences (left: LCC+LMLO, right: RCC+RMLO)"
    echo "    * Model 6: 2 inferences (LMLO, RMLO)"
    echo ""
    echo "=========================================="
    echo "System Information:"
    echo "=========================================="
    
    # System info
    echo "Hostname: $(hostname)"
    echo "CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
    echo "CPU Cores: $(nproc)"
    
    # Memory info
    echo ""
    echo "Memory Information:"
    free -h
    
    # Check if CUDA is available through Python
    echo ""
    echo "CUDA Status:"
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "Could not check CUDA status"
    
    echo ""
    echo "=========================================="
    echo "Starting Integrated Inference Test"
    echo "=========================================="
    echo ""
    
    # Record start time
    START_SECONDS=$(date +%s.%N 2>/dev/null || date +%s)
    
    # Run the Python inference script and capture all output
    python test_single.py \
        --rcc_image "$RCC_IMAGE" \
        --lcc_image "$LCC_IMAGE" \
        --rmlo_image "$RMLO_IMAGE" \
        --lmlo_image "$LMLO_IMAGE" \
        --device cuda \
        --normalization_flag "False" \
        --output_dir "$OUTPUT_DIR" \
        --seg_model_path "$SEG_MODEL_PATH" \
        --seg_img_size 256 2>&1
    
    PYTHON_EXIT_CODE=$?
    
    # Record end time
    END_SECONDS=$(date +%s.%N 2>/dev/null || date +%s)
    
    echo ""
    echo "=========================================="
    echo "Test Results"
    echo "=========================================="
    
    # Calculate total time
    if command -v python &> /dev/null; then
        TOTAL_TIME=$(python -c "print(f'{$END_SECONDS - $START_SECONDS:.3f}')")
        echo "Total Script Execution Time: ${TOTAL_TIME} seconds"
    fi
    
    echo "Python Exit Code: $PYTHON_EXIT_CODE"
    
    # Check system memory after
    echo ""
    echo "Post-Inference Memory State:"
    free -h
    
    echo ""
    if [ $PYTHON_EXIT_CODE -eq 0 ]; then
        echo "Status: SUCCESS - Inference completed"
        
        # Check if results were created
        if ls ${OUTPUT_DIR}/2d_proc.*.csv &>/dev/null 2>&1; then
            echo ""
            echo "Generated Files:"
            ls -lh ${OUTPUT_DIR}/2d_proc.*.csv 2>/dev/null
            ls -lh ${OUTPUT_DIR}/2d_proc.*.timing.csv 2>/dev/null
            
            # Display CSV contents
            echo ""
            echo "Results Preview:"
            echo "-----------------"
            if [ -f "${OUTPUT_DIR}/2d_proc.${IDENTIFIER}.csv" ]; then
                echo "Main results (first 5 lines):"
                head -5 "${OUTPUT_DIR}/2d_proc.${IDENTIFIER}.csv"
                
                echo ""
                echo "Total rows in results: $(wc -l < ${OUTPUT_DIR}/2d_proc.${IDENTIFIER}.csv)"
            fi
        fi
    else
        echo "Status: FAILED (Exit code: $PYTHON_EXIT_CODE)"
    fi
    
    echo ""
    echo "=========================================="
    echo "End Time: $(date)"
    echo "=========================================="
    
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Compliance Summary"
echo "=========================================="

# Read timing from CSV if available
if [ -f "${OUTPUT_DIR}/2d_proc.${IDENTIFIER}.timing.csv" ]; then
    # Extract timing info (assuming CSV format: header then values)
    INFER_TIME=$(awk -F',' 'NR==2{print $1}' "${OUTPUT_DIR}/2d_proc.${IDENTIFIER}.timing.csv")
    TOTAL_OPS=$(awk -F',' 'NR==2{print $2}' "${OUTPUT_DIR}/2d_proc.${IDENTIFIER}.timing.csv")
    REQUIREMENT_MET=$(awk -F',' 'NR==2{print $6}' "${OUTPUT_DIR}/2d_proc.${IDENTIFIER}.timing.csv")
    
    echo "- Inference time: ${INFER_TIME}s (target: <5s)"
    echo "- Total operations: ${TOTAL_OPS}"
    
    if [ "$REQUIREMENT_MET" == "True" ]; then
        echo "- Timing requirement: PASSED ✅"
    else
        echo "- Timing requirement: FAILED ❌"
    fi
fi

# GPU memory check
echo ""
echo "- GPU memory: Check Python output above (<16 GB required)"

# System RAM total
TOTAL_RAM=$(free -h | awk '/^Mem:/ {print $2}')
USED_RAM=$(free -h | awk '/^Mem:/ {print $3}')
echo "- System RAM: ${USED_RAM} used of ${TOTAL_RAM} total (<128 GB required)"

# Final message
echo ""
echo "=========================================="
echo "Complete log saved to: $LOG_FILE"
echo "Results saved to: ${OUTPUT_DIR}/2d_proc.${IDENTIFIER}.csv"
echo "Timing saved to: ${OUTPUT_DIR}/2d_proc.${IDENTIFIER}.timing.csv"
echo "=========================================="