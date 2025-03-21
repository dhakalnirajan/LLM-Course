#!/bin/bash

# Train the bigram model.
# Save the trained model to ../model.pth

# Change to the directory containing the Python script.  This is robust
# to where the script is called from.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR" || exit

# Train and save the model
python ../src/bigram.py --train ../data/input.txt --save ../model.pth