import torch.nn
import param
import torch.nn.utils.spectral_norm as spectral_norm


class ResBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels, type=None, scale=False):
		super(ResBlock, self).__init__()
		self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.last = None
		self.type = type
		self.scale = scale
		if type == 'gen':
			self.scaling_func = torch.nn.Upsample(scale_factor=2)
			modules = []
			modules.append(torch.nn.BatchNorm2d(in_channels))
			modules.append(torch.nn.ReLU())
			modules.append(torch.nn.Upsample(scale_factor=2))
			modules.append(self.conv1)
			modules.append(torch.nn.BatchNorm2d(out_channels))
			modules.append(torch.nn.ReLU())
			modules.append(self.conv2)
			self.model = torch.nn.Sequential(*modules)
		elif type == 'disc':
			self.scaling_func = torch.nn.AvgPool2d(2, stride=2, padding=0)
			if scale == False:
				modules = []
				modules.append(torch.nn.ReLU())
				modules.append(spectral_norm(self.conv1))
				modules.append(torch.nn.ReLU())
				modules.append(spectral_norm(self.conv2))
				self.model = torch.nn.Sequential(*modules)
			elif scale == True and in_channels == 3:
				modules = []
				modules.append(spectral_norm(self.conv1))
				modules.append(torch.nn.ReLU())
				modules.append(spectral_norm(self.conv2))
				modules.append(torch.nn.AvgPool2d(2, stride=2, padding=0))
				self.model = torch.nn.Sequential(*modules)
			elif scale == True and in_channels != 3:
				modules = []
				modules.append(torch.nn.ReLU())
				modules.append(spectral_norm(self.conv1))
				modules.append(torch.nn.ReLU())
				modules.append(spectral_norm(self.conv2))
				modules.append(torch.nn.AvgPool2d(2, stride=2, padding=0))
				self.model = torch.nn.Sequential(*modules)
		if in_channels != out_channels:
			self.last = spectral_norm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

	def shortcut(self, x):
		if self.type == 'gen': # Firstly apply upsampling, then conv
			if self.scale == True:
				x = self.scaling_func(x)
			if self.last:
				x = self.last(x)
		else: # Firstly apply conv, then downsampling
			if self.last:
				x = self.last(x)
			if self.scale == True:
				x = self.scaling_func(x)
		return x
	
	def forward(self, x):
		x_model = self.model(x)
		x_bypass = self.shortcut(x)
		return x_model + x_bypass

class SNGAN_G(torch.nn.Module):
	def __init__(self):
		super(SNGAN_G, self).__init__()
		self.dense = torch.nn.Linear(param.z_size, 4 * 4 * 64 * 16)
		self.final = torch.nn.Conv2d(64, param.n_colors, 3, stride=1, padding=1)
		main = torch.nn.Sequential()
		channels = [16, 16, 8, 4, 2, 1]
		for block_id in range(5):
			main.add_module('ResBlock_{}'.format(block_id), ResBlock(in_channels=64*channels[block_id], out_channels=64*channels[block_id + 1], type='gen', scale=True))
		main.add_module('BN', torch.nn.BatchNorm2d(64))
		main.add_module('ReLU', torch.nn.ReLU())
		main.add_module('FinalConv', self.final)
		main.add_module('Tanh', torch.nn.Tanh())

		self.main = main

	def forward(self, input):
		output = self.main(self.dense(input.view(input.shape[0], -1)).view(-1, 64*16, 4, 4))
		return output


# ResNet discriminator
class SNGAN_D(torch.nn.Module):
    def __init__(self):
        super(SNGAN_D, self).__init__()
        main = torch.nn.Sequential()
        rf = torch.nn.Sequential()
        jigsaw = torch.nn.Sequential()
        channels = [1, 2, 4, 8, 16]
        main.add_module('ResBlock_0', ResBlock(in_channels=3, out_channels=64, type='disc', scale=True))
        for block_id in range(1, 5):
            main.add_module('ResBlock_{}'.format(block_id), ResBlock(in_channels=64*channels[block_id-1], out_channels=64*channels[block_id], type='disc', scale=True))
        main.add_module('ResBlock_5', ResBlock(in_channels=64*16, out_channels=64*16, type='disc', scale=False))
        main.add_module('ReLU', torch.nn.ReLU())
        main.add_module('AvgPool', torch.nn.AvgPool2d(4))

        rf.add_module('Linear', spectral_norm(torch.nn.Linear(64*16, 1)))
        jigsaw.add_module('Linear', spectral_norm(torch.nn.Linear(64*16, 30)))
        self.main = main
        self.rf = rf
        self.jigsaw = jigsaw

    def forward(self, input):
        x = self.main(input).view(-1, 64*16)
        jigsaw = self.jigsaw(x)
        rf = self.rf(x)
        return rf.view(input.shape[0], -1), jigsaw.view(input.shape[0], -1)
