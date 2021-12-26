num_updates = 1 # meta-parameter update epochs, not used in this demo
thresh = 0.35 # threshold
lens = 0.5 # hyper-parameter in the approximate firing functions
decay = 0.4  # the decay constant of membrane potentials
num_classes = 10
batch_size = 100
num_epochs = 30
tau_w = 40 # synaptic filtering constant
lp_learning_rate = 5e-4  # learning rate of meta-local parameters
gp_learning_rate = 1e-3 # learning rate of gp-based parameters
time_window = 10 # time windows, we set T = 8 in our paper
w_decay = 0.95 # weight decay factor
cfg_fc = [512, 10] # Network structure