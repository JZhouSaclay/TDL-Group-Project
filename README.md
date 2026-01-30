# TDL-Group-Project
Deep Learning Scaling Laws: Feature Learning vs NTK
This repository contains a course project investigating the scaling laws of deep learning models. It specifically compares the generalization performance of standard finite-width neural networks (feature learning regime) against wide neural networks and the Neural Tangent Kernel (NTK) regime.

## Project Overview
The goal of this project is to analyze how the test error (Mean Squared Error) scales with the number of training samples (n) under different training regimes.

The project implements and compares:

Finite Width NN: Standard neural networks that perform feature learning.

Wide NN: Very wide networks initialized to approach the NTK limit.

Analytical NTK: Exact calculation of the infinite-width network kernel.

Empirical NTK: Kernel computed from the gradients of the network at initialization.

## Files
TDL_project.ipynb: The main Jupyter Notebook containing all code for data generation, model definitions, training loops, NTK computation, and plotting results.

## Key Implementations
Models: Custom PyTorch modules for FiniteWidthNN and WideNN.

NTK Tools: Functions to compute both Empirical NTK (using Jacobian computation) and Analytical NTK (for ReLU networks).

Data: Synthetic data generation using a Teacher-Student setup and polynomial target functions.

Visualization: Scripts to plot scaling law curves (Test MSE vs. Number of Samples).

## Requirements
The project requires Python 3 and the following libraries:

PyTorch

NumPy

SciPy

Matplotlib

Scikit-learn

## Usage
To reproduce the experiments and results:

Ensure all dependencies are installed.

Open TDL_project.ipynb in Jupyter Notebook or Google Colab (better try with GPU).

Run all cells to execute the training loops and generate the scaling laws plot.

## Group members: Jie ZHOU, Marine Vieilliard
