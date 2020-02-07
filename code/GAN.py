### Packages:
import os
import numpy as np
import math
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

### Initial Arguments: 
opt_n_epochs = 160 # number of epochs of training
opt_batch_size = 128 # size of the batches

opt_lr = 0.0002 # adam: learning rate
opt_b1 = 0.5 # adam: decay of first order momentum of gradient
opt_b2 = 0.999 # adam: decay of first order momentum of gradient

opt_latent_dim = 100 # dimensionality of the latent space
opt_channels = 1 # number of image channels
opt_img_size = 28  # size of each image dimension
opt_sample_interval = 2000 # interval betwen image samples

img_shape = (opt_channels, opt_img_size, opt_img_size)

#### Generator:
class Generator(nn.Module):
  def __init__(self, g_input_dim = opt_latent_dim, g_output_dim = np.prod(img_shape)):
    super(Generator, self).__init__() 
    self.g_input_dim = g_input_dim
    self.g_output_dim = g_output_dim

    def add_layer(in_feat, out_feat, normalize=False):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.ReLU(inplace=True))
        return layers
    
    self.G = nn.Sequential(
        *add_layer(self.g_input_dim,256),
        *add_layer(256,512),
        *add_layer(512,1024),
        nn.Linear(1024,self.g_output_dim),
        nn.Sigmoid()
    )

  def forward(self,z):
    fake_img = self.G(z)
    fake_img = fake_img.view(fake_img.size(0), *img_shape)
    return fake_img

#### Discrinimator:
class Discriminator(nn.Module):
    def __init__(self,d_input_dim = np.prod(img_shape)):
        super(Discriminator, self).__init__()
        self.d_input_dim = d_input_dim

        def add_layer(in_feat, out_feat,drop_rate,normalize=False):
          layers = [nn.Linear(in_feat, out_feat),nn.ReLU(inplace=True),nn.Dropout(drop_rate)]
          if normalize:
              layers.append(nn.BatchNorm1d(out_feat, 0.8))
          return layers

        self.D = nn.Sequential(
            *add_layer(self.d_input_dim, 1024, 0.25),
            *add_layer(1024, 512, 0.25),
            *add_layer(512, 256, 0.25),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.D(img_flat)
        return validity

# Loss Function:
adversarial_loss = torch.nn.BCELoss()

# Initialize Generator and Discriminator
generator = Generator()
discriminator = Discriminator()

### Cuda Available?:
cuda = True if torch.cuda.is_available() else False
print("Cuda is Available: ", cuda)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt_lr, betas=(opt_b1, opt_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt_lr, betas=(opt_b1, opt_b2))

# Load Data: 
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt_img_size), transforms.ToTensor()] #transforms.Normalize([0.5], [0.5]
        ),
    ),
    batch_size=opt_batch_size,
    shuffle=True,
)

# Save Loss:
loss_g, loss_d = [], []
def plot_loss(loss_g,loss_d,window,rolling_true=True):
    if rolling_true == True:
      D_ma = pd.Series(loss_d)
      G_ma = pd.Series(loss_g)
      loss_d_ma = list(D_ma.rolling(window).mean())
      loss_g_ma = list(G_ma.rolling(window).mean())
    plt.figure(figsize=(10,8))
    plt.plot(loss_d, label="Discriminator loss",color="#F8766D",linewidth=1)
    plt.plot(loss_g, label="Generator loss",color="#00BFC4",linewidth=1)
    plt.plot(loss_g_ma,color="#007d80",linewidth=2,linestyle="dashed")
    plt.plot(loss_d_ma,color="#b3554f",linewidth=2,linestyle="dashed")
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig("GAN_model_loss.png")
    plt.close()


# Save Images: 
def save_5by5(gen_images,epoch,h,w):
    # n images
    num_gen = 25
    
    # plot of generation
    n = np.sqrt(num_gen).astype(np.int32)
    I_generated = np.empty((h*n, w*n))
    for i in range(n):
        for j in range(n):
            I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = gen_images[i*n+j].cpu()

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(I_generated, cmap='gray')
    plt.show()
    plt.savefig("gan_imgs/%d.png" % epoch)
    plt.close    

# ----------
#  Training GAN: 
# ----------

for epoch in range(opt_n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt_latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt_n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        loss_g.append(g_loss.item())      
        loss_d.append(d_loss.item())

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt_sample_interval == 0:
            save_5by5(gen_imgs.data,batches_done,opt_img_size,opt_img_size)
            #save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

plot_loss(loss_g,loss_d,100)

#torch.save(generator)
#torch.save(generator.state_dict(),"/home/shlongenbach/")
file_name = "GAN_gpu_z100.pth"
torch.save(generator.state_dict(), file_name)
#torch.save(model, file_name)