### Instructions
We provide an example code to demonstrate the performance and robustness of three different learning models on the Fashion-MNIST datasets.

### How to run
Please load this folder path and directly run the file starting with ‘main_*’.
All models adopt a fully-connected structure and same share the same parameters (see shared_parameters.py) for fair comparison.

### Expected time
It takes about 30 s to run one training epoch with the one core of GTX 1080Ti.

### Expected results
After training, it can obtain the following performance:
LP model: 75.8%,
GP model: 87.9%,
HP model: 88.3%,

In the robustness exp. (e.g.noise level = 6),it can obtain the following performance:
LP model: 69.2%,
GP model: 65.5%,
HP model: 72.2%,

Please note that the results of the robust experiment will vary due to the random noise, but it does not affect the main conclusions of this experiment。
