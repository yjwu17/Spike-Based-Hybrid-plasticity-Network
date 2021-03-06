# Spiking-hybrid-plasticity-neural-network
- Our code will be coming soon. 
- Hybrid plasticity (HP) models provides a generic framework for training global-local hybrid SNNs using pytorch.
- HP model is designed to support multiple spike coding methods (rate-based and temporal based), multiple neuron models, and learning rules (Hebbian-based, STDP-based etc.)
- **Note**: If the environment configurations are different, the results may fail to work properly.
You may need to modify the package version or adjust code details according to your needs.

## Setup
All the codes of this project have been debugged and passed on Python 3.5.4 and Pycharm platforms. 
- Installation link for Pycharm: https://www.jetbrains.com/pycharm/download/
- Choose your operating system and Python version 3.5
- Download and install

## Requirements

Linux: Ubuntu 16.04

Cuda 9.0 & cudnn6.0

NVIDIA Titan Xp and NVIDIA GTX 1080. 

torch 1.2.0

torchvision 0.2.2

numpy 1.17.2

scipy 1.2.1

scikit-image 0.15.0

## Instructions for use
- File names starting with ‘main_*’ can be run to reproduce the results in this paper.
  
## An example demo
We provide a simple example code to help you quickly run our model and compare with other single-learning models. 

How to run: please load the folder of "simple-example" and run the "main_*" functions. 

Expected run time: 30s for one epoch (GTX 1080, one core). 

Expected results are shown in the readme file in the ‘simple-example’ folder.
  
## Reference
1. Wu Y, Zhao R, Zhu J, et al. Brain-inspired global-local hybrid learning towards human-like intelligence[J]. arXiv preprint arXiv:2006.03226, 2020.
