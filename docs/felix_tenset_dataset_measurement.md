# TenSet Dataset Measurement Guide

This guide provides the essential steps for setting up Felix and measuring
programs to build a dataset.

## Prerequisites

- Linux system with NVIDIA GPU
- CUDA toolkit installed
- Git
- Internet connection

## 1. Environment Setup

### Install Miniconda

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create and activate the recommended conda environment
conda env create -f python/env.yaml
conda activate felix
```

### Setup CUDA Path (if needed)

Add CUDA to your PATH in `~/.bashrc`:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
source ~/.bashrc
```

## 2. Clone and Build Felix

### Clone Repository

```bash
# If you have not already cloned Felix:
git clone https://github.com/uiuc-arc/felix.git
cd felix
```

### Initialize Submodules

```bash
git submodule update --init --recursive --progress
```

### Configure Build

```bash
mkdir build
cp cmake/config.cmake build/
# Edit build/config.cmake as needed for your system, enable CUDA and LLVM
```

### Build Felix

```bash
cmake -DCMAKE_BUILD_TYPE=Release -B build/ -S .
cmake --build build -j"$(nproc)"
```

### Install Python Dependencies

```bash
python -m pip install -e ./python
```

## 3. Dataset Setup

### Download Dataset

```bash
# Install gdown for Google Drive downloads
pip install gdown

# Download the TenSet dataset
gdown --fuzzy https://drive.google.com/file/d/1jqHbmvXUrLPDCIqJIaPee_atsPc0ZFFK/view

# Setup dataset directory expected by Felix scripts
mkdir -p lightning_logs/tenset
mv dataset_gpu_v3.3.zip lightning_logs/tenset/
cd lightning_logs/tenset/
unzip dataset_gpu_v3.3.zip
mv dataset_gpu/* .
rm -r dataset_gpu dataset_gpu_v3.3.zip
cd ../..
```

Expected directory structure after setup:

```text
lightning_logs/tenset/
|-- network_info/
|   `-- all_tasks.pkl
|-- to_measure_programs/
|   `-- *.json files
`-- measure_records/  (will be created)
```

## 4. Program Measurement

### Basic Usage

```bash
# Activate conda environment
conda activate felix

# Measure programs (uses parameters hard-coded in the script main())
python3 python/scripts/tenset_measure_programs.py
```

To customize task groups, number of tasks/configurations, or GPU devices, edit
`params` in `python/scripts/tenset_measure_programs.py`.
