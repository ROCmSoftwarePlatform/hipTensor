#!/bin/bash

rm -f merged_top10.txt

for i in `ls *.txt`
do
  sort -t ',' -n -k 2 $i | tail -n 5 >> merged_top10.txt
done

awk -F ',' '{counts[$1]++} END {for (item in counts) {print item "," counts[item]}}' merged_top10.txt | sort -t ',' -k 2nr | head -n 20 > final_top20.txt
