#!/usr/bin/env bash

# Get all C++ files checked into the repo under src/
root_folder=$(git rev-parse --show-toplevel)
all_files=$( \
    git ls-tree --full-tree -r --name-only HEAD . \
    | grep "src/.*\(\.cc\|\.h\|\.hpp\|\.cpp\|\.cu\)$" \
    | sed "s~^~$root_folder/~")
num_files=$(echo $all_files | wc -w)
echo "Formatting ${num_files} files"

cd $root_folder
clang-format -i --style=file $all_files
cd -