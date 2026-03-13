# Hybrid LLM Creation Using MergeKit (TinyLlama)

## Overview

This project demonstrates how to create a **hybrid Large Language Model (LLM)** by merging two open-source models using **MergeKit**.
The experiment was performed locally on a consumer laptop GPU to explore practical LLM engineering workflows such as model merging, dependency management, and local inference.

The merged model combines the behavior of two TinyLlama models to produce a new hybrid model capable of generating natural language responses.

---

## Objectives

The goal of this experiment was to:

* Understand how LLM weights can be merged
* Learn practical LLM engineering workflows
* Run local inference using merged models
* Build a reproducible experiment using open-source tools

---

## Models Used

Two open-source models were used:

* TinyLlama 1.1B Chat
* TinyLlama 1.1B Base

Both models share the same architecture, which allows safe parameter merging.

---

## Merge Strategy

The models were merged using **MergeKit** with a **linear weighted merge**.

Weight distribution:

TinyLlama Chat → 60%
TinyLlama Base → 40%

This approach combines conversational ability with general reasoning capability.

---

## Architecture

TinyLlama Chat Model
  +
TinyLlama Base Model
  ↓
MergeKit Linear Merge
  ↓
Hybrid TinyLlama Model
  ↓
Local Inference using PyTorch

---

## Hardware Setup

Laptop: MSI Katana 15
CPU: Intel i7-12650H
GPU: NVIDIA RTX 3050 (6GB)
RAM: 16GB
Operating System: Ubuntu Linux

---

## Tools and Libraries

* Python
* PyTorch
* MergeKit
* HuggingFace Transformers
* Safetensors
* Accelerate

---

## Installation

Clone the repository:

```
git clone https://github.com/halovivek/tinyllama-llm-merge-experiment.git
cd tinyllama-llm-merge-experiment
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Download the Models

Download the TinyLlama models from HuggingFace:

```
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 models/tinyllama_chat
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 models/tinyllama_base
```

---

## Run the Merge

Execute the MergeKit configuration:

```
mergekit-yaml merge.yml merged-model --copy-tokenizer
```

This creates a new merged model in the `merged-model` directory.

---

## Test the Model

Run the inference script:

```
python test_model.py
```

Example prompt:

```
Explain quantum computing in simple terms
```

Example output:

```
Quantum computers are machines that can perform complex calculations using the principles of quantum mechanics. They use quantum bits (qubits) which can exist in multiple states simultaneously, allowing them to solve certain problems more efficiently than classical computers.
```

---

## Project Structure

```
tinyllama-llm-merge-experiment
│
├── merge.yml
├── test_model.py
├── demo.py
├── requirements.txt
├── README.md
└── screenshots
```

Large model files are excluded from the repository using `.gitignore`.

---

## Key Learnings

This experiment provided hands-on experience with:

* LLM model merging
* Local LLM inference
* Managing ML dependencies
* HuggingFace model workflows
* Debugging PyTorch and Transformers environments

---

## Future Work

Future improvements may include:

* Creating a Mixture-of-Experts style multi-model system
* Integrating Retrieval Augmented Generation (RAG)
* Building a financial AI assistant for analyzing stock market data
* Developing a FinGPT-style assistant for the NSE market

---

## Author

Vivek Rajagopalan

---

## License

This project is for research and educational purposes.
