#!/bin/bash
# This is the first pipeline, as we assume you have already conda in your computer
echo "please make sure you have GPU larger than 24GvRAM for 7B model, 40GvRAM for 14B model, linux system"

export CONDA_ENV_NAME="py312torch230"
echo @CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.12


# open conda env
eval "$(conda shell.bash hook)"

conda activate $CONDA_ENV_NAME

# requirements
conda install cuda-cudart=12.1.105=0 -c nvidia

conda install pytorch=2.3.0=py3.12_cuda12.1_cudnn8.9.2_0  -c pytorch

pip install ninja
pip install flash-attn --no-build-isolation
pip install modelscope==1.18.0  # For model download
pip install openai==1.46.0
pip install tqdm==4.66.2
pip install transformers==4.44.2
pip install vllm==0.6.1.post2


# model download
modelscope download Qwen/Qwen2.5-14B-Instruct
modelscope download Qwen/Qwen2.5-7B-Instruct