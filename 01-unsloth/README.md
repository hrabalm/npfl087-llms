# Unsloth

## Environment creation

Open `screen` so that you don't lose access to your interactive job. You can detach with `Ctrl-a d` and reattach using `screen -r`.

You could also use `tmux` instead if you prefer to and know how to.

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

### Install miniforge3 (this includes mamba - a faster alternative to conda) to the current node's home directory

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### Create a conda prefix and install the necessary packages

```bash
TMPDIR="$SCRATCHDIR" ~/miniforge3/bin/mamba create --prefix=~/envs/unsloth pytorch cuda-nvcc=12.1 torchvision torchaudio pytorch-cuda=12.1 xformers -c pytorch -c nvidia -c xformers
```

### Install pip packages and unsloth

```bash
TMPDIR="$SCRATCHDIR" ~/miniforge3/bin/mamba run --prefix=~/envs/unsloth --no-capture-output python -m pip install --no-cache-dir bitsandbytes "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git@dd72d9f" wandb jupyterlab ipython sacrebleu
```

### Test installation

```bash
qsub -q gpu -l select=1:ncpus=1:ngpus=1:mem=8gb:scratch_local=16gb:cl_adan=True -I -l walltime=4:00:00
```

```bash
~/miniforge3/bin/mamba run --prefix=~/envs/unsloth --no-capture-output ipython
```

```python
import unsloth
```

### Jupyter

Modify environment variables in the attached shell script. Most importantly the path to the mamba executable, the conda environment and your home directory.

You can then run it with

```bash
qsub JupyterLabConda_Job.sh
```

You can list your jobs with `qstat -u [username]` and cancel the job early with `qdel [job id]`.

or you can run the job interactively (remember to use `screen` or `tmux` to prevent random disconnects):

```bash
qsub -I JupyterLabConda_Job.sh
./JupyterLabConda_Job.sh  # or the absolute path if in different home.
```

I had trouble connecting to the running instance with VS Code because of self-signed certificate errors. See [https://github.com/microsoft/vscode-jupyter/issues/7558#issuecomment-1450042948](https://github.com/microsoft/vscode-jupyter/issues/7558#issuecomment-1450042948) for a possible solution.
