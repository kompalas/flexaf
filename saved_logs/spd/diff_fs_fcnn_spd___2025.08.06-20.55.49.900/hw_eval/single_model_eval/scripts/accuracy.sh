#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 file1.txt file2.txt"
    exit 1
fi

file1=$1
file2=$2

# Check if files exist
if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
    echo "One or both files do not exist."
    exit 1
fi

# Read files and compare line by line
total=0
correct=0

while IFS= read -r line1 && IFS= read -r line2 <&3; do
    ((total++))
    if [ "$line1" -eq "$line2" ]; then
        ((correct++))
    fi
done < "$file1" 3< "$file2"

# Compute accuracy
if [ "$total" -eq 0 ]; then
    echo "Files are empty."
    exit 1
fi

accuracy=$(awk "BEGIN {print ($correct/$total) * 100}")
echo "Accuracy: $accuracy%"