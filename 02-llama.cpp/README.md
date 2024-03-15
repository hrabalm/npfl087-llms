# llama-cpp-python

## Environment creation

Interactive job

```bash
qsub -l select=1:ncpus=1:mem=8gb:scratch_local=16gb:cl_tarkil=True -I -l walltime=4:00:00
```

```bash
module add python/python-3.10.4-gcc-8.3.0-ovkjwzd cuda/cuda-11.2.0-intel-19.0.4-tn4edsz cudnn/cudnn-8.1.0.77-11.2-linux-x64-intel-19.0.4-wx22b5t
```

Create venv

```bash
export TMPDIR=$SCRATCHDIR
python3 -m venv venvs/llama-cpp-python
```

Install `llama-cpp-python` and other dependencies.

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" ~/venvs/llama-cpp-python/bin/pip install --no-cache-dir llam
a-cpp-python wandb sacrebleu jupyterlab ipython
```

Note that it is compiled with cublas backend (there might still be a CPU fallback).

To use another backend, for example for CPU inference, the library would have to be reinstalled and recompiled, so probably the easiest thing to do is to just create another venv, for example: `llama-cpp-python-cpu`. (just removing `CMAKE_ARGS` during installation should work fine)

```bash
export TMPDIR=$SCRATCHDIR
python3 -m venv venvs/llama-cpp-python-cpu
~/venvs/llama-cpp-python-cpu/bin/pip install --no-cache-dir llama-cpp-python wandb sacrebleu jupyterlab ipython
```

For more information see the official `llama-cpp-python` docs:
[https://llama-cpp-python.readthedocs.io/en/latest/](https://llama-cpp-python.readthedocs.io/en/latest/)

## Test

```bash
qsub -q gpu -l select=1:ncpus=1:ngpus=1:mem=8gb:scratch_local=16gb:cl_adan=True -I -l walltime=0:30:00
```

```bash
~/venvs/llama-cpp-python/bin/ipython
```

```python
from llama_cpp import Llama
llm = Llama("/storage/praha1/home/hrabalm/models/mistral-7b-v0.1.Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=4096)
```

## Translation

```python
prompt = 'English: I am sorry to hear that.\nCzech: To je mi líto.</s>\n\nEnglish: How much does it cost?\nCzech: Kolik to stojí?</s>\n\nEnglish: Prague is the capital of the Czech Republic.\nCzech: Praha je hlavní město České republiky.</s>\n\nEnglish: Pay attention to the road.\nCzech: Dávej pozor na silnici.</s>\n\nEnglish: I have a headache.\nCzech: Bolí mě hlava.</s>\n\nEnglish: How do you say this in Czech?\nCzech:"
print(prompt)
```

### Simple prompt

```python
print(llm(prompt, max_tokens=256))
```

### Stop tokens

```python
print(llm(prompt, max_tokens=256, stop=["\n"]))
```

## Other things to consider

### Batched inference

Implemented in llama.cpp, the author of `llama-cpp-python` claims he's been working on porting it for a while:

[https://github.com/abetlen/llama-cpp-python/issues/771](https://github.com/abetlen/llama-cpp-python/issues/771)
[https://github.com/abetlen/llama-cpp-python/pull/951](https://github.com/abetlen/llama-cpp-python/pull/951)

I am planning on creating a small benchmark to see if the performance gain is noticeable.

### Speculative decoding

`llama-cpp-python` implements a simple n-gram model which could speed up inference.
