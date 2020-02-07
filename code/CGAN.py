### Packages:
import os
import numpy as np
import pandas as pd
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
opt_n_epochs = 100 # number of epochs of training
opt_batch_size = 128 # size of the batches

opt_lr = 0.0002 # adam: learning rate
opt_b1 = 0.5 # adam: decay of first order momentum of gradient
opt_b2 = 0.999 # adam: decay of first order momentum of gradient

opt_latent_dim = 100 # dimensionality of the latent space
opt_channels = 1 # number of image channels
opt_img_size = 28  # size of each image dimension
opt_n_classes = 10 # unique number of labels
opt_sample_interval = 2000 # interval betwen image samples

img_shape = (opt_channels, opt_img_size, opt_img_size)

#### Generator:
class Generator(nn.Module):
    def __init__(self,g_input_dim = opt_latent_dim, g_output_dim = np.prod(img_shape),n_classes = opt_n_classes):
        super(Generator, self).__init__()

        self.g_input_dim = g_input_dim
        self.g_output_dim = g_output_dim
        self.n_classes = n_classes
        self.label_emb = nn.Embedding(n_classes,n_classes)

        def add_layer(in_feat, out_feat,neg_slope,normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(neg_slope, inplace=True))
            return layers
        
        self.G = nn.Sequential(
            *add_layer(self.g_input_dim + self.n_classes, 128, 0.2, normalize=False),
            *add_layer(128, 256, 0.2),
            *add_layer(256, 512, 0.2),
            *add_layer(512, 1024, 0.2),
            nn.Linear(1024, self.g_output_dim),
            nn.Sigmoid() #nn.Tanh()
        )

    def forward(self, z, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        fake_img = self.G(gen_input)
        fake_img = fake_img.view(fake_img.size(0), *img_shape)
        return fake_img


#### Discrinimator:
class Discriminator(nn.Module):
    def __init__(self, d_input_dim = np.prod(img_shape), n_classes = opt_n_classes):
        super(Discriminator, self).__init__()
        
        self.d_input_dim = d_input_dim
        self.n_classes = n_classes
        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.D = nn.Sequential(
            nn.Linear(self.d_input_dim  + self.n_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        img_flat = img.view(img.size(0), -1)
        d_in = torch.cat((img_flat, self.label_embedding(labels)), -1)
        validity = self.D(d_in)
        return validity


# Loss Function:
adversarial_loss = torch.nn.MSELoss() #torch.nn.BCELoss()

# Initialize Generator and Discriminator:
generator = Generator()
discriminator = Discriminator()

### Cuda Available?:
cuda = True if torch.cuda.is_available() else False
print("Cuda is Available: ", cuda)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Optimizers:
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt_lr, betas=(opt_b1, opt_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt_lr, betas=(opt_b1, opt_b2))

# Load Data: 
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt_img_size), transforms.ToTensor()] # transforms.Normalize([0.5], [0.5])
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
    plt.savefig("CGAN_model_loss.png")
    plt.close()



# Save Images: 
def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt_latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "cgan_imgs/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt_n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        #batch_size = imgs.shape[0]
        # Adversarial ground truths
        valid = Variable(FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt_latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt_n_classes, imgs.shape[0])))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

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
            sample_image(n_row=10, batches_done=batches_done)

plot_loss(loss_g,loss_d,100)

file_name = "CGAN_gpu_z100.pth"
torch.save(generator.state_dict(), file_name)