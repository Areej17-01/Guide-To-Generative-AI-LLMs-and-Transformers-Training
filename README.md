# Guide to Generative AI: LLMs and Transformers Training

Welcome to the **Guide to Generative AI: LLMs and Transformers Training** repository! This collection of Jupyter notebooks and scripts is designed to help you master the training, fine-tuning, and evaluation of large language models (LLMs) and transformer architectures using a variety of modern tools and techniques, including Axolotl, Hugging Face Transformers, LoRA, quantization, and more.

Whether you are a researcher, practitioner, or enthusiast, this repository provides hands-on, practical examples for training state-of-the-art models on custom datasets, optimizing for resource efficiency, and deploying your models for inference.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Notebooks and Their Purposes](#notebooks-and-their-purposes)
- [Axolotl: Fast, Flexible LLM Training](#axolotl-fast-flexible-llm-training)
- [LoRA and Quantization Techniques](#lora-and-quantization-techniques)
- [Environment Setup](#environment-setup)
- [Usage Guide](#usage-guide)
- [Best Practices & Tips](#best-practices--tips)
- [References](#references)

---

## Project Overview

This repository is a comprehensive guide to training and fine-tuning LLMs and transformer models. It covers:

- **Basic transformer concepts and tokenization**
- **Fine-tuning popular models (BERT, T5, GPT-2, Llama, etc.)**
- **Advanced training with Axolotl, including LoRA and quantization (4-bit, 8-bit)**
- **Custom dataset preparation and integration**
- **Efficient training on limited hardware (Colab, consumer GPUs)**
- **Model evaluation and inference**

---

## Notebooks and Their Purposes

### 1. `1) 6_LLM_Mastery_Transformer_section.ipynb`
- **Purpose:** Introduction to transformer models, tokenization, and the Hugging Face Transformers library.
- **Content:** Step-by-step exploration of BERT, tokenization, vocabulary, and encoding/decoding processes. Great for beginners to understand the building blocks of LLMs.

### 2. `2) T5_for_product_reviews.ipynb`
- **Purpose:** Fine-tuning T5 for product review summarization/classification.
- **Content:** Loads Amazon Reviews dataset, preprocesses data, and demonstrates end-to-end fine-tuning of T5 using Hugging Face's Trainer API. Includes data preparation, tokenization, and evaluation.

### 3. `3) section_7)_Bert_QnA_.ipynb`
- **Purpose:** Training and evaluating BERT for question-answering tasks.
- **Content:** Shows how to use BERT for extractive Q&A, including context/question encoding, answer span prediction, and visualization of token scores. Includes custom prediction functions and chunking for long contexts.

### 4. `4) 7_gpt_2_training.ipynb`
- **Purpose:** Fine-tuning GPT-2 (or similar decoder models) for language modeling.
- **Content:** Loads instruction-based datasets, preprocesses and tokenizes data, and demonstrates training with Hugging Face's Trainer. Includes batching, evaluation, and logging.

### 5. `5) simple_axolotl_training.ipynb`
- **Purpose:** Introduction to Axolotl for LLM training.
- **Content:** Shows how to install and configure Axolotl, prepare YAML configs, and launch training jobs. Demonstrates training with custom datasets and model parameters.

### 6. `6) Lora_training_with_llama_1b.ipynb`
- **Purpose:** Advanced LoRA (Low-Rank Adaptation) training with Llama-1B using Axolotl.
- **Content:** Covers LoRA configuration, YAML setup, and training launch. Explains LoRA parameters and their impact on training efficiency and memory usage.

### 7. `7)_LOAD_IN_8_BIT_TRAINING.ipynb`
- **Purpose:** Training LLMs in 8-bit precision for memory efficiency.
- **Content:** Shows how to configure Axolotl and Hugging Face Transformers for 8-bit training, including YAML config examples and launch commands.

### 8. `8)_train_in_4_bit_.ipynb`
- **Purpose:** Training LLMs in 4-bit precision (QLoRA) for maximum efficiency.
- **Content:** Demonstrates 4-bit quantization, dataset preparation, YAML configuration, and training/merging LoRA adapters. Includes troubleshooting for quantization and tokenizer issues.

### 9. `9) train in 4-bit-cot`
- **Purpose:** Chain-of-Thought (CoT) reasoning with 4-bit quantized LLMs.
- **Content:** Prepares medical reasoning datasets in Alpaca format, configures QLoRA training, and demonstrates inference and evaluation on complex reasoning tasks.

### 10. `testing-the-generation.ipynb`
- **Purpose:** Inference and generation testing for trained models.
- **Content:** Shows how to load trained models, apply chat templates, and generate responses using Hugging Face pipelines. Useful for validating model outputs after training.

---

## Axolotl: Fast, Flexible LLM Training

[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) is a powerful open-source framework for training and fine-tuning LLMs. It supports:

- **LoRA/QLoRA adapters for efficient fine-tuning**
- **Deepspeed and FlashAttention for speed and memory savings**
- **Flexible YAML-based configuration**
- **Integration with Hugging Face models and datasets**
- **Support for 4-bit and 8-bit quantization**

This repo provides several example YAML configs and training scripts to get you started with Axolotl.

---

## LoRA and Quantization Techniques

- **LoRA (Low-Rank Adaptation):** Efficiently fine-tune large models by training only a small set of parameters (adapters), reducing memory and compute requirements.
- **QLoRA (Quantized LoRA):** Combines LoRA with 4-bit quantization for even greater efficiency, enabling training of large models on consumer GPUs.
- **8-bit/4-bit Training:** Reduces memory usage and speeds up training/inference, with minimal loss in model quality.

The provided notebooks demonstrate how to set up, configure, and launch LoRA/QLoRA training jobs, including merging adapters and troubleshooting common issues.

---

## Environment Setup

### Requirements

- Python 3.8+
- Jupyter Notebook or Google Colab
- NVIDIA GPU (T4 or better recommended for large models)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Datasets](https://github.com/huggingface/datasets)
- [Deepspeed](https://www.deepspeed.ai/) (for large-scale training)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) (for quantization)

### Installation

Most notebooks include installation cells for required packages. For local setup, run:

```bash
pip install transformers datasets axolotl[flash-attn,deepspeed] bitsandbytes
```

For quantized training, ensure your CUDA and PyTorch versions are compatible.

---

## Usage Guide

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Guide-To-Generative-AI-LLMs-and-Transformers-Training.git
   cd Guide-To-Generative-AI-LLMs-and-Transformers-Training
   ```

2. **Open a notebook in Jupyter or Colab.**
   - Each notebook is self-contained and includes setup, data loading, training, and evaluation steps.

3. **Modify YAML configs for Axolotl as needed.**
   - Adjust model, dataset, and training parameters to fit your use case.

4. **Run training and monitor outputs.**
   - Use provided scripts and commands to launch training jobs.
   - Check logs and outputs for progress and troubleshooting.

5. **Test your trained models.**
   - Use `testing-the-generation.ipynb` or similar scripts to validate model outputs.

---

## Best Practices & Tips

- **Use Google Colab Pro or a local GPU for best performance.**
- **Monitor VRAM usage when training large models or using quantization.**
- **Always validate your dataset format and preprocessing steps.**
- **Experiment with LoRA/QLoRA parameters for optimal results.**
- **Keep your dependencies up to date, but ensure compatibility (especially for quantization and Axolotl).**
- **Backup your trained models and configs regularly.**

---

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Axolotl Documentation](https://axolotl-ai-cloud.github.io/axolotl/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Deepspeed](https://www.deepspeed.ai/)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

---

## Acknowledgements

This repository was created and maintained by Areej Mehboob. Contributions, issues, and pull requests are welcome!

---

**Happy Training! 🚀**
