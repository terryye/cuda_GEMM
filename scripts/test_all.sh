#!/bin/bash

# Function to run tests for scripts matching a pattern in subfolders

# Change to the directory where this script is located
cd "$(dirname "$0")/../src"

type=$1
#if type is empty, set it to modal
if [[ -z $type ]]; then
    type="modal"
fi

#if type is not modal or local, print usage and exit
if [[ $type != "modal" && $type != "local" ]]; then
    echo "Usage: $0 [modal|local]"
    exit 1
fi

pattern="run.sh"
#if type is modal or empty , check if modal is installed
if [[ $type == "modal" ]]; then
    if ! command -v modal &> /dev/null; then   
        echo -e "\033[31m modal is not installed, skipping modal tests. \033[0m"
        return
    fi
else
    pattern="run_local.sh"  
fi


# Script to test all run_modal_* scripts in subfolders
# For each subfolder, run the run_modal_*.sh script and check if "Test Pass" is in the output
for dir in */; do
    if [ -d "$dir" ]; then
        if [ "$dir" == "util/" ]; then
            continue
        fi

        echo "Testing running $dir"
        cd "$dir"
        # Find the script matching the pattern
        script=$(ls $pattern 2>/dev/null | head -1)
        if [ -n "$script" ]; then
            echo "Running: ./$script, this may take a couple of minutes."
            # Run the script and capture output
            output=$(./$script 2>&1)
            # Check if output contains "FAILED" or no "PASSED"
            if echo "$output" | grep -q "FAILED" || ! echo "$output" | grep -q "PASSED"; then
                echo -e "$dir: \033[31m FAILED \033[0m"
                echo "Output was:"
                echo "$output"
                echo "---"
            else
                echo -e "$dir: \033[32m pass \033[0m"
            fi
        else
            echo "$dir: no script matching $pattern found"
        fi
        cd ..
    fi
done