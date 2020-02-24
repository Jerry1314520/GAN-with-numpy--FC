## Packages:
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

## Initial Parameters: 
opt_n_epochs = 100 # number of epochs of training
opt_batch_size = 128 # size of the batches

opt_lr = 0.0002 # adam: learning rate
opt_b1 = 0.6 # adam: decay of first order momentum of gradient
opt_b2 = 0.999 # adam: decay of first order momentum of gradient

opt_latent_dim = 100 # dimensionality of the latent space
opt_channels = 1 # number of image channels
opt_img_size = 28  # size of each image dimension
opt_sample_interval = 1000 # interval betwen image samples

img_shape = (opt_channels, opt_img_size, opt_img_size)

## Generator:
class Generator(nn.Module):
  def __init__(self, g_input_dim = opt_latent_dim, g_output_dim = np.prod(img_shape)):
    super(Generator, self).__init__() 
    self.g_input_dim = g_input_dim
    self.g_output_dim = g_output_dim

    def add_layer(in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat,0.8))
        layers.append(nn.ReLU(inplace=True))
        #layers.append(nn.Dropout(drop_rate))
        return layers
    
    self.G = nn.Sequential(
        *add_layer(self.g_input_dim,256,normalize=False),
        *add_layer(256,512),
        *add_layer(512,1024),
        nn.Linear(1024,self.g_output_dim),
        nn.Tanh()
    )

  def forward(self,z):
    fake_img = self.G(z)
    fake_img = fake_img.view(fake_img.size(0), *img_shape)
    return fake_img

## Discrinimator:
class Discriminator(nn.Module):
    def __init__(self,d_input_dim = np.prod(img_shape)):
        super(Discriminator, self).__init__()
        self.d_input_dim = d_input_dim

        def add_layer(in_feat, out_feat,drop_rate=0.2,normalize=False):
          layers = [nn.Linear(in_feat, out_feat), nn.ReLU(inplace=True), nn.Dropout(drop_rate)]
          if normalize:
              layers.append(nn.BatchNorm1d(out_feat, 0.8))
          return layers

        self.D = nn.Sequential(
            *add_layer(self.d_input_dim, 1024),
            *add_layer(1024, 512),
            *add_layer(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.D(img_flat)
        return validity

## Loss Function:
adversarial_loss = torch.nn.BCELoss()

## Initialize Generator and Discriminator
generator = Generator()
discriminator = Discriminator()

## Cuda Available?:
cuda = True if torch.cuda.is_available() else False
print("Cuda is Available: ", cuda)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

## Optimizers:
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt_lr, betas=(opt_b1, opt_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt_lr, betas=(opt_b1, opt_b2))

## Load Data: 
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt_img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])] #Scale -1 to 1
        ),
    ),
    batch_size=opt_batch_size,
    shuffle=True,
)

## Save Loss:
loss_g, loss_d, epoch_index = [], [], []
def plot_loss(loss_g,loss_d,window,rolling_true=True,epoch_index = None):
    if rolling_true == True:
      D_ma = pd.Series(loss_d)
      G_ma = pd.Series(loss_g)
      loss_d_ma = list(D_ma.rolling(window).mean())
      loss_g_ma = list(G_ma.rolling(window).mean())
    plt.figure(figsize=(18,6))
    file_name = "GANloss_.png"
    if epoch_index != None:
      file_name = "GANloss_%d.png" % epoch_index
      if epoch_index > window:
        plt.scatter(epoch_index,loss_g_ma[epoch_index],color="#c46700",s=100,zorder=1e2,alpha=0.8)
      else:
        plt.scatter(epoch_index,loss_g[epoch_index],color="#c46700",s=100,zorder=1e2,alpha=0.8)
    plt.plot(loss_d, label="Discriminator loss",color="#F8766D",linewidth=1)
    plt.plot(loss_g, label="Generator loss",color="#00BFC4",linewidth=1)
    plt.plot(loss_g_ma,color="#007d80",linewidth=2,linestyle="dashed")
    plt.plot(loss_d_ma,color="#b3554f",linewidth=2,linestyle="dashed")
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig("gan_losses/"+file_name,bbox_inches='tight') #
    plt.close()

## Save Images: 
def save_4by10(gen_images,epoch,h,w):
    # plot of generation
    height,width = 3,10
    I_generated = np.empty((h*height, w*width))
    for i in range(height):
        for j in range(width):
            I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = gen_images[i*height+j].cpu()

    plt.figure(figsize=(18, 6))
    plt.axis("off")
    I_generated*127.5 + 127.5
    plt.imshow(I_generated, cmap='gray')
    t = plt.text(1, 3, epoch, fontsize=10,color='black',fontweight='bold',backgroundcolor='0')
    t.set_bbox(dict(facecolor='#c46700', alpha=0.8, edgecolor='#c46700'))
    plt.show()
    plt.savefig("gan_imgs/4by10_%d.png" % epoch,bbox_inches='tight') #
    plt.close   

#---------------
## GAN Training: 
#---------------
for epoch in range(opt_n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Labels (real = 1 , fake = 0):
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Inputs (real images, generator noise):
        real_imgs = Variable(imgs.type(Tensor))
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt_latent_dim))))

        ## Train Generator:
        optimizer_G.zero_grad()
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        ## Train Discriminator:
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        ## Store Mini-Batch Loss:
        loss_g.append(g_loss.item())      
        loss_d.append(d_loss.item())

        #print(
        #    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #    % (epoch, opt_n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        #)

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt_sample_interval == 0:
            save_4by10(gen_imgs.data,batches_done,opt_img_size,opt_img_size)
            epoch_index.append(batches_done)

## Loss Plots:             
for e in epoch_index:
  plot_loss(loss_g,loss_d,window = 100,epoch_index=e)

## Save GIFs:
import imageio

# 4by10 GIF:
os.chdir("/home/shlongenbach/gan_imgs/")
images = []
for filename in ["4by10_"+str(d)+".png" for d in epoch_index]:
    images.append(imageio.imread(filename))
os.chdir("/home/shlongenbach/")
imageio.mimsave("GAN_4by10.gif", images,duration=0.2)

# Loss GIF:
os.chdir("/home/shlongenbach/gan_losses/")
images = []
for filename in ["GANloss_"+str(d)+".png" for d in epoch_index]:
    images.append(imageio.imread(filename))
os.chdir("/home/shlongenbach/")
imageio.mimsave("GAN_loss.gif", images,duration=0.2)

## Save Model:
file_name = "GAN_gpu_z100.pth"
torch.save(generator.state_dict(), file_name)

