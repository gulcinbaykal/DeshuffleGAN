### REFERENCE: https://github.com/AlexiaJM/RelativisticGAN 
import torch.nn
import param

class DCGAN_G(torch.nn.Module):
    def __init__(self):
        super(DCGAN_G, self).__init__()
        main = torch.nn.Sequential()

        # We need to know how many layers we will use at the beginning
        mult = param.image_size // 8

        ### Start block
        # Z_size random numbers
        main.add_module('Start-ConvTranspose2d', torch.nn.ConvTranspose2d(param.z_size, param.G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False))
        main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(param.G_h_size * mult))
        main.add_module('Start-ReLU', torch.nn.ReLU())
        # Size = (G_h_size * mult) x 4 x 4

        ### Middle block (Done until we reach ? x image_size/2 x image_size/2)
        i = 1
        while mult > 1:
            main.add_module('Middle-ConvTranspose2d [%d]' % i, torch.nn.ConvTranspose2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
            main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(param.G_h_size * (mult//2)))
            main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU())
            # Size = (G_h_size * (mult/(2*i))) x 8 x 8
            mult = mult // 2
            i += 1

        ### End block
        # Size = G_h_size x image_size/2 x image_size/2
        main.add_module('End-ConvTranspose2d', torch.nn.ConvTranspose2d(param.G_h_size, param.n_colors, kernel_size=4, stride=2, padding=1, bias=False))
        main.add_module('End-Tanh', torch.nn.Tanh())
        # Size = n_colors x image_size x image_size
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output

# DCGAN discriminator (using somewhat the reverse of the generator)
class DCGAN_D(torch.nn.Module):
    def __init__(self):
        super(DCGAN_D, self).__init__()
        main = torch.nn.Sequential()
        normal = torch.nn.Sequential()
        jigsaw = torch.nn.Sequential()

        ### Start block
        # Size = n_colors x image_size x image_size
        main.add_module('Start-Conv2d', torch.nn.Conv2d(param.n_colors, param.D_h_size, kernel_size=4, stride=2, padding=1, bias=False))
        main.add_module('Start-LeakyReLU', torch.nn.LeakyReLU(0.2, inplace=True))
        image_size_new = param.image_size // 2
        # Size = D_h_size x image_size/2 x image_size/2

        ### Middle block (Done until we reach ? x 4 x 4)
        mult = 1
        i = 0
        while image_size_new > 4:
            main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(param.D_h_size * mult, param.D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False))
            main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(param.D_h_size * (2*mult)))
            main.add_module('Middle-LeakyReLU [%d]' % i, torch.nn.LeakyReLU(0.2, inplace=True))
            # Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
            image_size_new = image_size_new // 2
            mult *= 2
            i += 1

        ### End block
        # Size = (D_h_size * mult) x 4 x 4
        normal.add_module('End-Conv2d', torch.nn.Conv2d(param.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
        jigsaw.add_module('End_Conv2d', torch.nn.Conv2d(param.D_h_size * mult, 30, kernel_size=4, stride=1, padding=0, bias=False))

        if param.loss_D == 1:
            normal.add_module('End-Sigmoid', torch.nn.Sigmoid())
            jigsaw.add_module('End-Sigmoid', torch.nn.Sigmoid())
        
        # Size = 1 x 1 x 1 (Is a real cat or not?)
        self.main = main
        self.normal = normal
        self.jigsaw = jigsaw

    def forward(self, input):
        x = self.main(input)
        jigsaw = self.jigsaw(x)
        rf = self.normal(x)
        # Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
        return rf.view(input.shape[0], -1), jigsaw.view(input.shape[0], -1)
