#!/bin/bash

set -ex

SCRIPT_DIR=$(dirname "$0")

ls *.txt | xargs sed -i -z "s/There are 52 instances.\n//"

sh $SCRIPT_DIR/merge_top10.sh

sh $SCRIPT_DIR/search_first_top_in_most_frequent_driver.sh | sort -t '_' -n -k 2 -k 3 -k 4 -k 5  > best_instances.txt

python3 $SCRIPT_DIR/convert_to_cpp_pair.py

