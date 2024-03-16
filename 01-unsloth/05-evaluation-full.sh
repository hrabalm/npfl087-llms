#!/bin/bash
#FIXME PBS...

echo ${PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}

PYTHON=/storage/praha1/home/hrabalm/miniforge3/bin/python
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )  # source: https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
FILE="$SCRIPT_DIR/05-evaluation-full.py"

"$PYTHON" "$FILE"
clean_scratch
