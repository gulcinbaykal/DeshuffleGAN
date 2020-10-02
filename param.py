image_size = 128
batch_size = 64 
n_colors = 3
z_size = 128
# Number of hidden nodes in the Generator.
G_h_size = 128  
# Number of hidden nodes in the Discriminator.
D_h_size = 128
# Discriminator learning rate
lr_D = .0002 
#Generator learning rate
lr_G = .0002
# Number of iteration cycles
n_iter = 200000
# Adam betas[0]
beta1 = 0
# Adam betas[1]
beta2 = 0.9
# Loss choices are 1=standard gan loss (SGAN), 2=least square loss (LSGAN), 3=hinge loss (HingeGAN), 4=relativistic average standard gan loss (RaSGAN), 5=relativistic average least square loss (RaLSGAN), 6= relativistic average hinge loss(RaHingeGAN)
loss_D = 3
# Baseline choices are DCGAN (1) or SNGAN (2). For SNGAN, loss_D=3 is used
arch = 2
# Decay to apply to lr each cycle
decay = 0
seed = 1
# Dataset folder
input_folder = '/okyanus/users/gbaykal/MyGAN/input_celeb/'
# Output folder to save your results
output_folder = '/okyanus/users/gbaykal/MyGAN/output'
# Model is to save or not
save = True
# Path to the network you want to load
load = None
cuda = True
# Number of Discriminator iterations
Diters = 2
# Number of Generator iterations
Giters = 1
# L2 regularization weight
weight_decay = 0
# Generate at the end of each gen_every iterations
gen_every = 10000
# Number of generated samples at the end of each gen_every iterations
gen_extra_images = 10000
# Generate a mini-batch of images at every print_every iterations to see how the training progress
print_every = 1000
# Decide if the deshuffling is included or not
jigsaw = True
# Influence of the deshuffling objective for the Discriminator objective (alpha)
d_weight = 1.0
# Influence of the deshuffling objective for the Generator objective (beta)
g_weight = 0.5
