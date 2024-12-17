#!/bin/bash

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname $SCRIPT_PATH)

# echo $SCRIPT_DIR/search_first_top_in_most_frequent.py $i final_top20.txt

for i in `ls F*.txt`
do
  python3 $SCRIPT_DIR/search_first_top_in_most_frequent.py $i final_top20.txt
done
