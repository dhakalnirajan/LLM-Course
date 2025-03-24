#!/bin/bash

# Train the micrograd model.

# Change to the directory containing the Python script.  This is robust
# to where the script is called from.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR" || exit

# Train and save the model (you may not use it for this chapter, but it is a good habit).
python ../notebooks/micrograd_example.ipynb #running the notebook.