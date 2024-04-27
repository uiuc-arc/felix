# Felix, the Gradient-based Optimizer for Tensor Programs

Felix is a gradient-based compiler optimization framework for tensor-based programs.
It is designed to find optimization schedule in short amount of time
by using gradient descent on the schedule parameters directly.

This repository is the official implementation of <br>
[**Felix: Optimizing Tensor Programs with Gradient Descent**](https://dl.acm.org/doi/10.1145/3620666.3651348) <br>
(Yifan Zhao, Hashim Sharif, Vikram Adve, Sasa Misailovic; ASPLOS 2024).

## Installation

Felix is based on the tensor compiler framework TVM;
this repository contains a patched version of TVM.
Check out the [installing TVM](https://tvm.apache.org/docs/tutorial/install.html)
guide for the prerequisites and installation steps;
these installation steps also apply to this repository.

- In addition to TVM's prerequisites, Felix requires a system-wide Rust Cargo installation,
  which you can find ways to install [here](https://github.com/rust-lang/cargo).
- The following setup may work the best and is recommended:
  - Linux. Windows may work but Felix has never been tested on Windows.
  - CUDA 11 and above, with a [version-compatible](https://gist.github.com/ax3l/9489132) C++ compiler;
  - A Python virtual environment with PyTorch installed.
    If you use Conda as your virtual-env manager,
    you can build an environment using the env file [`./python/env.yaml`](python/env.yaml).

## Getting Started

Once the setup finishes, you can import Felix in Python as `tvm.felix`.
Check out the [example file in gallery](gallery/tutorial/felix_tune_network.py)
for instructions on how to run Felix and optimize a DNN.
