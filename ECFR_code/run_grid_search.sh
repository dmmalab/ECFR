cfr#!/bin/bash

# ============================================================================
# Oracle Quasi-Logarithmic Grid Search Protocol for Fair Comparison
# Search Space: {1, 3.3, 6.6} x {10^-5, 10^-4} U {10^-3}
# ============================================================================

DATASET_CSV="path/to/your/dataset.csv"
METHOD="ecfr" # Options: ecfr, grata

# The exact quasi-logarithmic learning rate list specified in the paper
LR_LIST=(1e-5 3.3e-5 6.6e-5 1e-4 3.3e-4 6.6e-4 1e-3)

echo "Starting Oracle Grid Search for method: ${METHOD^^}"
echo "Search space: ${LR_LIST[*]}"
echo "---------------------------------------------------------"

for lr in "${LR_LIST[@]}"; do
    echo "[ Grid Search ] Running ${METHOD^^} with LR = $lr"
    
    # Execute the adaptation script
    python main_adaptation.py \
        --dataset_csv ${DATASET_CSV} \
        --method ${METHOD} \
        --lr ${lr} \
        --capacity 128 \
        --quantile 0.5
        
    echo "Finished LR = $lr"
    echo "---------------------------------------------------------"
done

echo "Grid Search Completed. Please check the logs for the best performing LR (Oracle selection)."
