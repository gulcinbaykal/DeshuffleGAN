### REFERENCE: This code is written mainly based on https://github.com/AlexiaJM/RelativisticGAN . Thanks !
import os
import numpy
import torch
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transf
import torchvision.models as models
import torchvision.utils as vutils
import torch.nn.utils.spectral_norm as spectral_norm
from archs.dcgan import DCGAN_G, DCGAN_D
from archs.sngan import SNGAN_G, SNGAN_D

from utils import Shuffler

import param

import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True

from IPython.display import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import math

torch.utils.backcompat.broadcast_warning.enabled=True

# Setting the title: Firstly, set the baseline architecture
if param.arch == 1:
	title = 'DCGAN_'
if param.arch == 2:
	title = 'SNGAN_'

# Setting the title: Secondly, set the baseline version original or DeshuffleGAN
if param.jigsaw == True:
	title = title + 'deshuffle_'
else:
	title = title + 'original_'

# Setting the title: Lastly, set the seed value which is always used as 1
if param.seed is not None:
	title = title + 'seed%i' % param.seed


# Check folder run-i for all i=0,1,... until it finds run-j which does not exists, then creates a new folder run-j

run = 0
base_dir = f"{param.output_folder}/{title}-{run}"
if param.load:
	run = int(param.load.split('/')[-4].split('-')[-1])
	base_dir = f"{param.output_folder}/{title}-{run}"
else:
	while os.path.exists(base_dir):
		run += 1
		base_dir = f"{param.output_folder}/{title}-{run}"
os.makedirs(base_dir, exist_ok=True)
logs_dir = f"{base_dir}/logs"
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(f"{base_dir}/images", exist_ok=True)
if param.gen_extra_images > 0 and not os.path.exists(f"{base_dir}/extra"):
	os.makedirs(f"{base_dir}/extra", exist_ok=True)

# where we save the output
if param.load:
	log_output = open(f"{logs_dir}/log.txt", 'a')
else:
	log_output = open(f"{logs_dir}/log.txt", 'w')
param_list = []
for item in vars(param).items():
	if not item[0].startswith('__'):
		param_list.append(item)

print(param_list)
print(param_list, file=log_output)

## Setting seed
import random
if param.seed is None:
	param.seed = random.randint(1, 10000)
print(f"Random Seed: {param.seed}")
print(f"Random Seed: {param.seed}", file=log_output)
random.seed(param.seed)
numpy.random.seed(param.seed)
torch.manual_seed(param.seed)
if param.cuda:
	torch.cuda.manual_seed_all(param.seed)

# Transforming images
trans = transf.Compose([
	transf.Resize((param.image_size, param.image_size)),
	# This makes it into [0,1]
	transf.ToTensor(),
	# This makes it into [-1,1]
	transf.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])
# You can chance this setting if your dataset can be loaded from torchvision.datasets
data = dset.ImageFolder(root=param.input_folder, transform=trans)

# Loading data randomly
def generate_random_sample():
	while True:
		random_indexes = numpy.random.choice(data.__len__(), size=param.batch_size, replace=False)
		batch = [data[i][0] for i in random_indexes]
		yield torch.stack(batch, 0)
random_sample = generate_random_sample()

# Model initialization
if param.arch == 1:
	G = DCGAN_G()
	D = DCGAN_D()
else:
	G = SNGAN_G()
	D = SNGAN_D()
# Initialize weights
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		# Estimated variance, must be around 1
		m.weight.data.normal_(1.0, 0.02)
		# Estimated mean, must be around 0
		m.bias.data.fill_(0)
G.apply(weights_init)
D.apply(weights_init)
print("Initialized weights")
print("Initialized weights", file=log_output)

# Criterion
criterion = torch.nn.BCELoss()
BCE_stable = torch.nn.BCEWithLogitsLoss()
jigsaw_criterion = torch.nn.CrossEntropyLoss()

# Soon to be variables
x = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size)
x_fake = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size)
y = torch.FloatTensor(param.batch_size, 1)
y2 = torch.FloatTensor(param.batch_size, 1)
z = torch.FloatTensor(param.batch_size, param.z_size, 1, 1)
# Uniform weight
u = torch.FloatTensor(param.batch_size, 1, 1, 1)
# This is to see during training, size and values won't change
z_test = torch.FloatTensor(param.batch_size, param.z_size, 1, 1).normal_(0, 1)

# Everything cuda
if param.cuda:
	G = G.cuda()
	D = D.cuda()
	criterion = criterion.cuda()
	BCE_stable.cuda()
	jigsaw_criterion.cuda()
	x = x.cuda()
	x_fake = x_fake.cuda()
	y = y.cuda()
	y2 = y2.cuda()
	u = u.cuda()
	z = z.cuda()
	z_test = z_test.cuda()

# Now Variables
x = Variable(x)
x_fake = Variable(x_fake)
y = Variable(y)
y2 = Variable(y2)
z = Variable(z)
z_test = Variable(z_test)

optimizerD = torch.optim.Adam(D.parameters(), lr=param.lr_D, betas=(param.beta1, param.beta2), weight_decay=param.weight_decay)
optimizerG = torch.optim.Adam(G.parameters(), lr=param.lr_G, betas=(param.beta1, param.beta2), weight_decay=param.weight_decay)

# exponential weight decay on lr
decayD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=1-param.decay)
decayG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=1-param.decay)

# Load existing models
if param.load:
	checkpoint = torch.load(param.load)
	current_set_images = checkpoint['current_set_images']
	iter_offset = checkpoint['i']
	G.load_state_dict(checkpoint['G_state'])
	D.load_state_dict(checkpoint['D_state'])
	optimizerG.load_state_dict(checkpoint['G_optimizer'])
	optimizerD.load_state_dict(checkpoint['D_optimizer'])
	decayG.load_state_dict(checkpoint['G_scheduler'])
	decayD.load_state_dict(checkpoint['D_scheduler'])
	z_test.copy_(checkpoint['z_test'])
	del checkpoint
	print(f'Resumed from iteration {current_set_images*param.gen_every}.')
else:
	current_set_images = 0
	iter_offset = 0

print(G)
print(G, file=log_output)
print(D)
print(D, file=log_output)

## Fitting model
for i in range(iter_offset, param.n_iter):

	# Fake images saved
	if i % param.print_every == 0:
		fake_test = G(z_test)
		vutils.save_image(fake_test.data, '%s/images/fake_samples_iter%05d.png' % (base_dir, i), normalize=True)

	for p in D.parameters():
		p.requires_grad = True

	for t in range(param.Diters):

		########################
		# (1) Update D network #
		########################

		D.zero_grad()
		images = random_sample.__next__()
		# Mostly necessary for the last one because if N might not be a multiple of batch_size
		current_batch_size = images.size(0)
		if param.cuda:
			images = images.cuda()
		if param.jigsaw == True:
			real_data_shuffler = Shuffler(images, 30, resize=param.image_size)
			shuffled_real_data, real_data_shuffle_orders = real_data_shuffler.shuffle()
			del real_data_shuffler
			shuffled_real_data = shuffled_real_data.cuda()
			real_data_shuffle_orders = torch.LongTensor(real_data_shuffle_orders)
			real_data_shuffle_orders = real_data_shuffle_orders.cuda()
		# Transfer batch of images to x
		x.data.resize_as_(images).copy_(images)
		del images
		y_pred, _ = D(x)

		if param.jigsaw == True:
			_, D_real_jigsaw = D(shuffled_real_data)
			disc_jigsaw_loss = jigsaw_criterion(D_real_jigsaw, real_data_shuffle_orders)
		
		if param.loss_D in [1,2,3]: # Relativistic least squares loss
			y.data.resize_(current_batch_size, 1).fill_(1)
			if param.loss_D == 1:
				errD_real = criterion(y_pred, y)
			if param.loss_D == 2:
				errD_real = torch.mean((y_pred - y) ** 2)
			if param.loss_D == 3:
				errD_real = torch.mean(torch.nn.ReLU()(1.0 - y_pred))
			
			# Train with fake data
			z.data.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
			fake = G(z)
			x_fake.data.resize_(fake.data.size()).copy_(fake.data)
			y.data.resize_(current_batch_size, 1).fill_(0)
			# Detach y_pred from the neural network G and put it inside D
			y_pred_fake, _ = D(x_fake.detach())
			if param.loss_D == 1: #SGAN
				errD_fake = criterion(y_pred_fake, y)
			if param.loss_D == 2: #LSGAN
				errD_fake = torch.mean((y_pred_fake) ** 2)
			if param.loss_D == 3: #HingeGAN
				errD_fake = torch.mean(torch.nn.ReLU()(1.0 + y_pred_fake))

			errD = errD_real + errD_fake
			if param.jigsaw:
				errD = errD + param.d_weight * disc_jigsaw_loss
			errD.backward()

		else:
			y.data.resize_(current_batch_size, 1).fill_(1)
			y2.data.resize_(current_batch_size, 1).fill_(0)
			z.data.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
			fake = G(z)
			x_fake.data.resize_(fake.data.size()).copy_(fake.data)
			y_pred_fake, _ = D(x_fake.detach())
			if param.loss_D == 4: #RaSGAN
				errD = (BCE_stable(y_pred - torch.mean(y_pred_fake), y) + BCE_stable(y_pred_fake - torch.mean(y_pred), y2))/2
			if param.loss_D == 5: #RaLSGAN
				errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) + y) ** 2))/2
			if param.loss_D == 6: #RaHingeGAN
				errD = (torch.mean(torch.nn.ReLU()(1.0 - (y_pred - torch.mean(y_pred_fake)))) + torch.mean(torch.nn.ReLU()(1.0 + (y_pred_fake - torch.mean(y_pred)))))/2
			errD_real = errD
			errD_fake = errD
			if param.jigsaw:
				errD = errD + param.d_weight * disc_jigsaw_loss				
			errD.backward()		

		optimizerD.step()


	########################
	# (2) Update G network #
	########################

	# Make it a tiny bit faster
	for p in D.parameters():
		p.requires_grad = False

	for t in range(param.Giters):

		G.zero_grad()
		y.data.resize_(current_batch_size, 1).fill_(1)
		z.data.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
		fake = G(z)
		y_pred_fake, _ = D(fake)		

		if param.jigsaw == True:
			generated_data_shuffler = Shuffler(fake, 30, resize=param.image_size)
			shuffled_fake_data, fake_data_shuffle_orders = generated_data_shuffler.shuffle()
			del generated_data_shuffler
			shuffled_fake_data = shuffled_fake_data.cuda()
			fake_data_shuffle_orders = torch.LongTensor(fake_data_shuffle_orders)
			fake_data_shuffle_orders = fake_data_shuffle_orders.cuda()
		
		if param.loss_D not in [1, 2, 3]:
			images = random_sample.__next__()
			current_batch_size = images.size(0)
			if param.cuda:
				images = images.cuda()
			x.data.resize_as_(images).copy_(images)
			del images

		if param.loss_D == 1:
			errG = criterion(y_pred_fake, y)
		if param.loss_D == 2:
			errG = torch.mean((y_pred_fake - y) ** 2)
		if param.loss_D == 3:
			errG = -torch.mean(y_pred_fake)
		if param.loss_D == 4:
			y_pred, _ = D(x)
			# Non-saturating
			y2.data.resize_(current_batch_size, 1).fill_(0)
			errG = (BCE_stable(y_pred - torch.mean(y_pred_fake), y2) + BCE_stable(y_pred_fake - torch.mean(y_pred), y))/2
		if param.loss_D == 5:
			y_pred, _ = D(x)
			errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) - y) ** 2))/2
		if param.loss_D == 6:
			y_pred, _ = D(x)
			# Non-saturating
			errG = (torch.mean(torch.nn.ReLU()(1.0 + (y_pred - torch.mean(y_pred_fake)))) + torch.mean(torch.nn.ReLU()(1.0 - (y_pred_fake - torch.mean(y_pred)))))/2
		if param.jigsaw == True:
			_, D_fake_jigsaw = D(shuffled_fake_data)
			gen_jigsaw_loss = jigsaw_criterion(D_fake_jigsaw, fake_data_shuffle_orders)
			errG = errG + param.g_weight * gen_jigsaw_loss			
		errG.backward()
		optimizerG.step()
	decayD.step()
	decayG.step()

	# Log results 

	if (i+1) % param.print_every == 0:
		if param.jigsaw:
			fmt = '[%d] loss_D: %.4f loss_G: %.4f disc_jigsaw_loss: %.4f gen_jigsaw_loss: %.4f gan_D: %.4f gan_G: %.4f alpha*disc_jigsaw_loss: %.4f beta*gen_jigsaw_loss: %.4f'
			s = fmt % (i, errD.data.item(), errG.data.item(), disc_jigsaw_loss.data.item(), gen_jigsaw_loss.data.item(), errD.data.item()-param.d_weight*disc_jigsaw_loss.data.item(), errG.data.item()-param.g_weight*gen_jigsaw_loss.data.item(), param.d_weight*disc_jigsaw_loss.data.item(), param.g_weight*gen_jigsaw_loss.data.item())
		else:
			fmt = '[%d] loss_D: %.4f loss_G: %.4f'
			s = fmt % (i, errD.data.item(), errG.data.item())
		print(s)
		print(s, file=log_output)

	# Save and generate
	if (i+1) % param.gen_every == 0:

		current_set_images += 1

		# Save models
		if param.save:
			if not os.path.exists('%s/models/' % (f"{base_dir}/extra")):
				os.makedirs('%s/models/' % (f"{base_dir}/extra"), exist_ok=True)
			torch.save({
				'i': i + 1,
				'current_set_images': current_set_images,
				'G_state': G.state_dict(),
				'D_state': D.state_dict(),
				'G_optimizer': optimizerG.state_dict(),
				'D_optimizer': optimizerD.state_dict(),
				'G_scheduler': decayG.state_dict(),
				'D_scheduler': decayD.state_dict(),
				'z_test': z_test,
			}, '%s/models/state_%02d.pth' % (f"{base_dir}/extra", current_set_images))
			s = 'Models saved'
			print(s)
			print(s, file=log_output)

		# Delete previously existing images
		if os.path.exists('%s/%01d/' % (f"{base_dir}/extra", current_set_images)):
			for root, dirs, files in os.walk('%s/%01d/' % (f"{base_dir}/extra", current_set_images)):
				for f in files:
					os.unlink(os.path.join(root, f))
		else:
			os.mkdir('%s/%01d/' % (f"{base_dir}/extra", current_set_images))

		# Generate images for FID/Inception to be calculated later 
		ext_curr = 0
		z_extra = torch.FloatTensor(100, param.z_size, 1, 1)
		if param.cuda:
			z_extra = z_extra.cuda()
		for ext in range(int(param.gen_extra_images/100)):
			fake_test = G(Variable(z_extra.normal_(0, 1)))
			for ext_i in range(100):
				vutils.save_image((fake_test[ext_i].data*.50)+.50, '%s/%01d/fake_samples_%05d.png' % (f"{base_dir}/extra", current_set_images,ext_curr), normalize=False, padding=0)
				ext_curr += 1
		del z_extra
		del fake_test
