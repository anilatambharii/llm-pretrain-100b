# llm-pretrain-100b
LLM Pretraining Framework (100B+ Params): Megatron-LM + DeepSpeed + FSDP. Open-source, HPC-ready system with tiny GPT simulation, distributed training, tokenizer tools, dataset pipelines, and deployment scripts for Slurm, AWS, Azure, and Docker.

# LLM Pretraining Framework (100B+ Parameters)

## Overview
This is an **open-source, production-grade framework** for pretraining **Large Language Models (LLMs)** up to 100B+ parameters.  
It includes:
- **Simulation mode** with a tiny GPT for local runs
- **Full HPC configurations** for Megatron-LM + DeepSpeed + PyTorch FSDP
- Tokenizer training
- Petabyte-scale data streaming structure
- Slurm, AWS, and Docker deployment scripts

---

## 🚀 Quick Start (Local Simulation)

```bash
git clone <your_repo_url>
cd llm-pretrain-100b
pip install -r requirements.txt

# Prepare a tiny dataset
bash scripts/prepare_dataset.sh

# Train tiny GPT locally
bash scripts/train_local.sh

## HPC Training (Slurm)
sbatch scripts/train_slurm.sh
Create Modules or Scripts for each feature:

distributed_training/ → DeepSpeed, Megatron-LM, FSDP integration

tokenizers/ → SentencePiece, HuggingFace support

data_pipeline/ → Petabyte-scale data handling

cloud_configs/ → AWS, Azure, GCP deployment scripts

parallelism/ → Hybrid parallelism logic

evaluation/ → Perplexity and zero-shot evaluation tools

