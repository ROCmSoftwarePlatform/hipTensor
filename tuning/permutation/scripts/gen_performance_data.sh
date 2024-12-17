#!/bin/bash

set -e
SCRIPT_DIR=$(dirname "$0")

# Function to handle the checks and renaming for a given rank
process_rank() {
  local rank="$1"

  # 1. Check if data{rank} exists and is not empty, then rename
  local data_dir="data${rank}"
  local timestamp=$(date +%Y%m%d%H%M%S)

  if [ -d "$data_dir" ] && [ "$(ls -A "$data_dir")" ]; then
    echo "Directory '$data_dir' exists and is not empty."
    local new_data_dir="${data_dir}_${timestamp}"
    mv "$data_dir" "$new_data_dir"
    if [ $? -eq 0 ]; then
        echo "Directory renamed to '$new_data_dir'."
    else
        echo "Error renaming directory. Exiting..."
        return 1 # Return non-zero to indicate failure
    fi
    mkdir $data_dir
  elif [ ! -d "$data_dir" ]; then
    echo "Directory '$data_dir' does not exist. Create the folder."
    mkdir $data_dir
  elif [ -d "$data_dir" ] && [ ! "$(ls -A "$data_dir")" ]; then
    echo "Directory '$data_dir' exists but is empty."
  fi

  # 2. Check if permutation_tuning_{rank} exists and run it
  local tuning_script="gen_performance_data${rank}.py"
  local tuning_exec="permutation_tuning_${rank}"
  if [ -x "$tuning_exec" ]; then # -x checks for executable
    echo "Running $tuning_script..."
	python3 "$SCRIPT_DIR/$tuning_script"
    if [ $? -ne 0 ]; then
        echo "Error running $tuning_script. Exiting..."
        return 1 # Return non-zero to indicate failure
    fi
  else
    echo "Script $tuning_script not found or not executable."
  fi

  return 0 # Return 0 to indicate success
}


# Process for each rank
for rank in 2 3 4; do
  echo "Processing rank: $rank"
  if ! process_rank "$rank"; then # If process_rank returns a non-zero exit status
    echo "Processing for rank $rank failed. Exiting..."
    exit 1 # Exit the main script
  fi
  echo "Finished processing rank: $rank"
  echo "---------------------"
done

echo "Script finished."
