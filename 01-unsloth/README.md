# Unsloth

## Environment creation

Open `screen` so that you don't lose access to your interactive job. You can detach with `Ctrl-a d` and reattach using `screen -r`. You could also use `tmux` if you prefer to and know how.

```bash
screen
```

### Ask for an interactive task (installation can take a while):

```bash
qsub -l select=1:ncpus=1:mem=8gb:scratch_local=16gb -I -l walltime=4:00:00
```

You can also specify the cluster - see [https://metavo.metacentrum.cz/pbsmon2/qsub_pbspro](https://metavo.metacentrum.cz/pbsmon2/qsub_pbspro)

```bash
qsub -l select=1:ncpus=1:mem=8gb:scratch_local=16gb:cl_adan=True -I -l walltime=4:00:00
```

### Install miniforge3 (this includes mamba - faster conda alternative) to the current node's home directory.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### Create a conda prefix and install the necessary packages

```bash
TMPDIR="$SCRATCHDIR "~/miniforge3/bin/mamba create --prefix=~/envs/unsloth pytorch cudatoolkit cudatoolkit-dev torchvision torchaudio pytorch-cuda=11.8 xformers -c pytorch -c nvidia -c xformers
```

### Install pip packages and unsloth

```bash
... python -m pip install --no-cache-dir bitsandbytes "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git@dd72d9f" wandb
```

### Test

```bash

```
