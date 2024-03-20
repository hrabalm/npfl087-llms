#!/bin/bash
#PBS -N llm_demo_evaluation_full
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=16gb:ngpus=1:gpu_cap=cuda70:cl_adan=True
#PBS -l walltime=24:00:00
#PBS -m ae

echo ${PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}

PYTHON=/storage/praha1/home/hrabalm/envs/unsloth/bin/python
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )  # source: https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script - broken on PBS, because the script is not run from the original directory
SCRIPT_DIR="$PBS_O_WORKDIR"
FILE="$SCRIPT_DIR/05-evaluation-full.py"

"$PYTHON" "$FILE"
clean_scratch
