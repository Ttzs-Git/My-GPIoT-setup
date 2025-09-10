# My-GPIoT-setup
## GPIoT Modified Setup for Local Reproduction & Enhancement

This repository contains my modifications, environment setup scripts, and documentation for reproducing and extending the **GPIoT** project.

## 📌 Original Project

**Title**: GPIoT: Tailoring Small Language Models for IoT Program Synthesis and Development  
**Authors**: Leming Shen, Qiang Yang, Xinyu Huang, Zijing Ma, Yuanqing Zheng  
**Conference**: ACM SenSys 2025  
**License**: MIT  
**Original Repo**: https://github.com/lemingshen/GPIoT   
**Paper**: https://lemingshen.github.io/assets/publication/conference/GPIoT/paper.pdf 
**Project**: [GPIoT: Tailoring Small Language Models for IoT Program Synthesis and   Development](https://lemingshen.github.io/projects/gpiot/)  

> ⚠️ This is NOT the official repository. I do not distribute the original model weights or datasets. Please download them from the authors' provided links.

## 🧩 My Contributions

- ✅ Provided a detailed dependency setup tailored to my hardware environment

- ✅ Modified **`task_generation.py`** to optimize task generation logic

- ✅ Modified **`code_generation.py`** to improve code generation workflow

- ✅ Modified **`fine_tune.py`** to enhance training stability and adaptability

  > These modifications improve local reproducibility, training stability, and code generation effectiveness.

## 📥 Installation

### ⚙️ My Setup Instructions

### Hardware Environment

- **GPU**: NVIDIA RTX 3090
- **Environment**: Conda virtual environment

### Key Packages

The project was tested under the following major dependencies:

| Package       | Version      |
| ------------- | ------------ |
| torch         | 2.1.0+cu118  |
| torchvision   | 0.16.0+cu118 |
| torchaudio    | 2.1.0+cu118  |
| transformers  | 4.36.0       |
| accelerate    | 0.25.0       |
| datasets      | 2.15.0       |
| peft          | 0.12.0       |
| trl           | 0.7.6        |
| bitsandbytes  | 0.41.3       |
| scipy         | 1.15.3       |
| pandas        | 2.3.2        |
| matplotlib    | 3.7.5        |
| wandb         | 0.21.3       |
| sentencepiece | 0.1.99       |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourname/your-gpiot-modified
cd your-gpiot-modified

# 2. Create and activate conda environment
conda create -n gpiot-env python=3.10 -y
conda activate gpiot-env

# 3. Install dependencies
pip install -r requirements.txt

```

