# Automated Date Extraction Pipeline
This repository contains the implementation of an automated date extraction system for French administrative documents, developed as part of the NLP in Industry final project at Université Paris Cité M2 NLP.

**Group Members:**
- Xu Sun
- Dan Hou
- Haaeul Hwang
- Léo Mégret

## Introduction
![Pipeline Schema](https://github.com/user-attachments/assets/eb89bc00-fc9d-41c1-b3c6-d063f7f00047)

Our pipeline aims to efficiently extract dates from French administrative documents through several key steps:

## Pipeline Components

### 1. Data Preprocessing (`1_dataset_rebuild.py`)
The first step focuses on preparing high-quality data for NER and LLM processing:

- **Asynchronous Data Collection**
  - Downloads content from provided URLs using aiohttp
  - Validates file accessibility and content length (>500 chars)
  - Handles parallel downloads efficiently

- **Text Normalization**
  - Concatenates source URL with document content
  - Removes redundant whitespace and special characters
  - Creates two versions of content:
      - `text_content`: URL + normalized text
      - `raw_text_content`: Only normalized text

- **Output**
- Generates `dataset_valid.csv` containing:
    - Original metadata
    - Local file paths
    - Normalized text content
    - Raw text content

### 2. Date Entity Extraction (`2_ner.py`)
Extracts candidate dates using NER:

- **Model**: Utilizes CamemBERT-NER (fine-tuned for French date extraction)

- **Optimization Features**:
  - GPU-accelerated batch processing
  - Dynamic text chunking for long documents
  - Efficient memory management
  - Multi-worker data loading

- **Processing Steps**:
  - Tokenizes and processes text in batches
  - Identifies date entities
  - Filters and validates dates
  - Maintains extraction context

### 3. LLM-based Date Selection (`4_llm_reference.py`)
Uses LLM reasoning to select the most accurate publication date:

- **Model**: Qwen2 5.7B/14B
  - Instruction tuned for natural language understanding
  - Enhanced context window (80k tokens) with the help of Vllm and rope-scaling

- **Selection Process**:
  - Takes NER extracted dates as candidates
  - Uses few-shot learning with carefully selected examples
  - Processes full document context
  - Returns single most likely publication date

- **Key Features**:
  - VLLM acceleration for fast inference
  - Efficient batching and memory usage
  - Robust error handling
  - Context-aware date selection

### 5. Date Format Cleaning (clean_date.py)
Standardizes extracted dates into a uniform format:

- **Date Pattern Recognition**

  - Handles multiple French date formats:

    * DD Month YYYY (e.g., "1er juillet 2023")
    * DD/MM/YYYY or DD/MM/YY
    * DD-MM-YYYY or DD-MM-YY
    * Month YYYY (e.g., "OCTOBRE 2022")


  - **Supports variations in month names (lowercase, uppercase, accented)**
  
  
  - **Format Standardization**

    * Converts all dates to DD/MM/YYYY format
    * Handles French month names using comprehensive mapping
    * Processes special cases like "1er" (first of month)
    * Assumes 20xx for two-digit years
    * Sets default day to 01 for month-year only dates

### 6. Evaluation (6_evaluation.py)
   Assesses the accuracy of date extraction results:

- **Accuracy Metrics**

  *   Compares extracted dates with gold standard labels
  *   Calculates two accuracy scores:
      1. Datapolitics accuracy (published vs. gold label)
      2. Our prediction accuracy (cleaned prediction vs. gold label)

  **Output Generation**
    *   Creates comprehensive evaluation report
    *   Includes metadata and comparison columns
    *   Saves results in CSV format


  - **Key Features**
    *   Handles missing columns gracefully
    *   Supports flexible input/output paths
    *   Provides formatted accuracy statistics


## Usage

### Environment Setup
please Run pipeline_environment.sh as it will create following environment for you (linux + conda only!)
Please use conda to create a virtual environment to avoid conflicts:

```bash
# Ubuntu 22.04 + CUDA 12.1
conda create -n automated_date_extraction python=3.12
conda activate automated_date_extraction

# Install PyTorch 2.3.0
conda install cuda-cudart=12.1.105=0 -c nvidia
conda install pytorch=2.3.0=py3.12_cuda12.1_cudnn8.9.2_0  -c pytorch

# Install required packages (specific versions required)
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

```
**Note**: Windows compatibility is not guaranteed due to flash_attn dependencies. For flash_attn installation (v2.6.3), please check the official release page: https://github.com/Dao-AILab/flash-attention


## Performance Benchmarks

| Stage | Time (NVIDIA 4090)   | Memory Usage                                        |
| ----- |----------------------|-----------------------------------------------------|
| Data Preprocessing | ~2 min               | 4GB RAM                                             |
| NER Processing | ~30 min              | 8GB VRAM                                            | 
| LLM Selection | ~1 min per inference | 23.2GB VRAM at least for 7B, 40 Gb at least for 14B |
| Total Pipeline | ~several hours       | 24GB VRAM peak                                      |

## Acknowledgments

### Models
- CamemBERT-NER by Jean-Baptiste: French NER model fine-tuned for date extraction
- Qwen2.5 series by Alibaba: Advanced LLMs optimized for reasoning tasks

### Data
- Datapolitics: Provided the French administrative document corpus

### Tools
- VLLM: High-performance LLM inference engine
- Flash Attention: Efficient attention computation