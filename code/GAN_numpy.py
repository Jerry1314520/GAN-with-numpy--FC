import numpy as np
import matplotlib.pyplot as plt
## Only 4 Loading Data: 
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class numpyGAN():
    def __init__(self, batch_size=64, noise_dim=100, img_size=28, learning_rate=0.001, decay_rate=0.0001, n_epochs=50, label_subset=[0,1,2,3,4,5,6,7,8,9]):

        ## Intial Parameters: 
        self.n_epochs = n_epochs
        self.batch_size = batch_size # size of the batches
        self.noise_dim = noise_dim # dimensionality of the latent space
        self.img_size = img_size  # size of each image dimension
        self.lr = learning_rate # learning rate
        self.dr = decay_rate 
        self.labels = label_subset

        ## Intial Generator Weights:
        self.G_W1 = np.random.randn(self.noise_dim, 128) * np.sqrt(2. / self.noise_dim) # 100 x 128
        self.G_b1 = np.zeros((1, 128)) # 1 x 128
        
        self.G_W2 = np.random.randn(128, 256) * np.sqrt(2. / 128) # 128 x 256
        self.G_b2 = np.zeros((1, 256)) # 1 x 256

        self.G_W3 = np.random.randn(256, self.img_size ** 2) * np.sqrt(2. / 256) # 256 x 784
        self.G_b3 = np.zeros((1, self.img_size ** 2)) # 1 x 784

        ## Intial Discriminator Weights:
        self.D_W1 = np.random.randn(self.img_size ** 2, 128) * np.sqrt(2. / self.img_size ** 2) # 784 x 128
        self.D_b1 = np.zeros((1, 128)) # 1 x 128

        self.D_W2 = np.random.randn(128, 1) * np.sqrt(2. / 128) # 128 x 1
        self.D_b2 = np.zeros((1, 1)) # 1 x 1
    
        def load_data():
            MNIST_data = datasets.MNIST(
                root='./data',
                train=True,
                download=True)
            # Subset of Numbers:
            mask = np.argwhere(np.isin(MNIST_data.targets,self.labels))
            MNIST_data_subset = MNIST_data.data[mask].type(torch.FloatTensor)
            # Scale -1 to 1: 
            mean,sd = 0.5,0.5
            MNIST_data_subset_transform = (MNIST_data_subset.div(255)-mean).div(sd)
            # Load into Batches:
            dataloader = torch.utils.data.DataLoader(dataset= MNIST_data_subset_transform,batch_size=self.batch_size,shuffle=True)
            return dataloader
        self.MNST_Data = load_data()
    
    ## Activation Functions:
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def tanh(self,z):
        return np.tanh(z)

    def lrelu(self,z,neg_slope):
        return np.maximum(z*neg_slope, z)

    def dsigmoid(self,a):
        da_dz = a*(1. - a)
        return da_dz

    def dtanh(self,a):
        da_dz = 1. - a**2
        return da_dz

    def dlrelu(self,a,neg_slope):
        da_dz = np.ones_like(a)
        da_dz[a < 0] = neg_slope
        return da_dz

    ## Loss Functions:
    def Gloss(self,d_ouput_fake):
        g_loss = -np.log(d_ouput_fake)
        return g_loss
    
    def dGloss(self,a, y="fake", eps=1e-8):
        if y =="fake":
            dL_da = -1.0 / (a + eps) 
        else:
            raise Exception('Select Label: (y = "fake")')
        return dL_da

    def Dloss(self,d_ouput_fake, d_output_real):
        d_loss_fake = -np.log(1 - d_ouput_fake)
        d_loss_real  = -np.log(d_output_real)
        d_loss = d_loss_fake + d_loss_real
        return d_loss
    
    def dDLoss(self,a, y="real", eps=1e-8 ):
        if y == "real":
            dL_da = -1. / (a + eps)
        elif y == "fake":
            dL_da = 1. / (1. - a + eps)
        else:
            raise Exception('Select Label: (y = "real" or "fake")')
        return dL_da

    ## Add Forward & Backward Layers:
    def hidden_layer_forward(self,x, W, b, acitvation='lrelu',ns=0.02):
        z = np.dot(x,W) + b
        if acitvation == 'lrelu':
            a = self.lrelu(z,ns)
        elif acitvation == 'sigmoid':
            a = self.sigmoid(z)
        elif acitvation == 'tanh':
            a = self.tanh(z)
        else:
            raise Exception('Non-supported activation function')
        return a

    def hidden_layer_backward(self,a, acitvation='lrelu',ns=0.02):
        if acitvation == 'lrelu':
            da_dz = self.dlrelu(a,ns)
        elif acitvation == 'sigmoid':
            da_dz = self.dsigmoid(a)
        elif acitvation == 'tanh':
            da_dz = self.dtanh(a)
        else:
            raise Exception('Non-supported activation function')
        return da_dz

    ## Generator:
    def generator_forward(self, noise):
        self.G_a1 = self.hidden_layer_forward(noise,self.G_W1,self.G_b1,acitvation='lrelu',ns=0)
        self.G_a2 = self.hidden_layer_forward(self.G_a1,self.G_W2,self.G_b2,acitvation='lrelu',ns=0)
        self.G_a3 = self.hidden_layer_forward(self.G_a2,self.G_W3,self.G_b3,acitvation='tanh')
        '''ADD MORE LAYERS'''
        return self.G_a3
    
    def generator_backward(self, noise, a_fake): 
        '''Discriminator Loss: '''
        dL_da2 = self.dGloss(a_fake, y="fake", eps=1e-8)
        da2_dz2 = self.hidden_layer_backward(a_fake, acitvation='sigmoid')
        dz2_da1 = self.D_W2.T
        da1_dz1 = self.hidden_layer_backward(self.D_a1, acitvation='lrelu')
        dz1_dx = self.D_W1.T

        dL_dx = np.dot(da1_dz1 * np.dot((da2_dz2 * dL_da2), dz2_da1), dz1_dx)

        '''Generator Loss: '''
        da3_dz3 = self.hidden_layer_backward(self.G_a3, acitvation='tanh')
        dz3_dW3 = self.G_a2.T
        dz3_db3 = np.ones(a_fake.shape[0])

        dL_dW3 = np.dot(dz3_dW3, (da3_dz3 * dL_dx))
        dL_db3 = np.dot(dz3_db3, (da3_dz3 * dL_dx))

        dz3_da2 = self.G_W3.T
        da2_dz2 = self.hidden_layer_backward(self.G_a2, acitvation='lrelu',ns=0)
        dz2_dW2 = self.G_a1.T
        dz2_db2 = np.ones(a_fake.shape[0])

        dL_dW2 = np.dot(dz2_dW2, da2_dz2 * np.dot( (da3_dz3 * dL_dx), dz3_da2))
        dL_db2 = np.dot(dz2_db2, da2_dz2 * np.dot( (da3_dz3 * dL_dx), dz3_da2))

        dz2_da1 = self.G_W2.T
        da1_dz1 = self.hidden_layer_backward(self.G_a1, acitvation='lrelu',ns=0)
        dz1_dW1 = noise.T
        dz1_db1 = np.ones(a_fake.shape[0])

        dL_dW1 = np.dot(dz1_dW1, da1_dz1 * np.dot( (da2_dz2 * np.dot( (da3_dz3 * dL_dx), dz3_da2)), dz2_da1))
        dL_db1 = np.dot(dz1_db1, da1_dz1 * np.dot( (da2_dz2 * np.dot( (da3_dz3 * dL_dx), dz3_da2)), dz2_da1))

        '''Update Generator Weights: '''
        self.G_W1 -= self.lr * dL_dW1 
        self.G_b1 -= self.lr * dL_db1 

        self.G_W2 -= self.lr * dL_dW2 
        self.G_b2 -= self.lr * dL_db2 

        self.G_W3 -= self.lr * dL_dW3 
        self.G_b3 -= self.lr * dL_db3 

        return None

    ## Discriminator:
    def discriminator_forward(self, img):
        self.D_a1 = self.hidden_layer_forward(img,self.D_W1,self.D_b1,acitvation='lrelu')
        self.D_a2 = self.hidden_layer_forward(self.D_a1,self.D_W2,self.D_b2,acitvation='sigmoid')
        '''ADD MORE LAYERS'''
        return self.D_a2

    def discriminator_backward(self, x_real, a_real, x_fake, a_fake):
        '''Discriminator Real Loss: '''
        dL_da2 = self.dDLoss(a_real, y="real", eps=1e-8) # 64x1
        da2_dz2 = self.hidden_layer_backward(a_real, acitvation='sigmoid') # a_real = self.D_a2
        dz2_dW2 = self.D_a1.T #64x128
        dz2_db2 = np.ones(a_real.shape[0]) #64x1

        dL_dW2_real = np.dot(dz2_dW2, (da2_dz2 * dL_da2))
        dL_db2_real = np.dot(dz2_db2, (da2_dz2 * dL_da2)) 

        dz2_da1 = self.D_W2.T
        da1_dz1 = self.hidden_layer_backward(self.D_a1, acitvation='lrelu')
        dz1_dW1 = x_real.T
        dz1_db1 = np.ones(a_real.shape[0])

        dL_dW1_real = np.dot(dz1_dW1, da1_dz1 * np.dot((da2_dz2 * dL_da2),dz2_da1) ) 
        dL_db1_real = np.dot(dz1_db1, da1_dz1 * np.dot((da2_dz2 * dL_da2),dz2_da1) ) 

        '''Discriminator Fake Loss: '''
        dL_da2 = self.dDLoss(a_fake, y="fake", eps=1e-8)
        da2_dz2 = self.hidden_layer_backward(a_fake, acitvation='sigmoid') # a_fake = self.D_a2
        dz2_dW2 = self.D_a1.T
        dz2_db2 = np.ones(a_fake.shape[0])

        dL_dW2_fake = np.dot(dz2_dW2, (da2_dz2 * dL_da2))
        dL_db2_fake = np.dot(dz2_db2, (da2_dz2 * dL_da2)) 

        dz2_da1 = self.D_W2.T
        da1_dz1 = self.hidden_layer_backward(self.D_a1, acitvation='lrelu',ns=0)
        dz1_dW1 = x_fake.T
        dz1_db1 = np.ones(a_fake.shape[0])

        dL_dW1_fake = np.dot(dz1_dW1, da1_dz1 * np.dot((da2_dz2 * dL_da2),dz2_da1) ) 
        dL_db1_fake = np.dot(dz1_db1, da1_dz1 * np.dot((da2_dz2 * dL_da2),dz2_da1) ) 

        '''Discriminator Combined Loss: '''
        dL_dW2_total = dL_dW2_real + dL_dW2_fake 
        dL_db2_total = dL_db2_real + dL_db2_fake

        dL_dW1_total = dL_dW1_real + dL_dW1_fake 
        dL_db1_total = dL_db1_real + dL_db1_fake

        '''Update Discriminator Weights: '''
        self.D_W1 -= self.lr * dL_dW1_total 
        self.D_b1 -= self.lr * dL_db1_total 

        self.D_W2 -= self.lr * dL_dW2_total 
        self.D_b2 -= self.lr * dL_db2_total 
        return None

    def save_5by5(self,gen_images,epoch,h,w):
        # n images
        num_gen = 25
        # plot of generation
        n = np.sqrt(num_gen).astype(np.int32)
        I_generated = np.empty((h*n, w*n))
        for i in range(n):
            for j in range(n):
                I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = gen_images[i*n+j]
        plt.figure(figsize=(5, 5))
        plt.axis("off")
        plt.imshow(I_generated, cmap='gray')
        #plt.savefig("%d.png" % epoch)
        plt.show()
        plt.close    
        return None

    def train(self):
        G_loss, D_loss = [], []
        files = []
        for epoch in range(self.n_epochs):
            for i, (imgs) in enumerate(self.MNST_Data):
                ## Only use Full Mini-Batches:
                if imgs.shape[0] != self.batch_size:
                  break
                ## Mini-Batch Real & Noise:
                real_imgs = imgs.numpy().reshape((imgs.shape[0], -1))
                noise = np.random.normal(0, 1, (imgs.shape[0], self.noise_dim))

                ## Forward:
                fake_imgs = self.generator_forward(noise)
                a_fake = self.discriminator_forward(fake_imgs)
                a_real = self.discriminator_forward(real_imgs)

                ## Track Loss:
                G_batch_loss = np.mean(self.Gloss(a_fake))
                D_batch_loss = np.mean(self.Dloss(a_fake, a_real))
                G_loss.append(G_batch_loss)
                D_loss.append(D_batch_loss)

                ## Backward:
                self.discriminator_backward(real_imgs,a_real,fake_imgs,a_fake)
                self.generator_backward(noise,a_fake)

            #Print Statistics:
            batches_done = epoch * len(self.MNST_Data) + i
            if epoch % 5 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [LR: %f]"
                    % (epoch, self.n_epochs, i, len(self.MNST_Data), D_batch_loss, G_batch_loss,self.lr)
                )
                gen_imgs = fake_imgs.reshape((self.batch_size,1,self.img_size,self.img_size))
                self.save_5by5(gen_imgs,epoch,self.img_size,self.img_size)
                files.append("%d.png" % epoch)
            #Update Learning Rate:
            self.lr = self.lr * (1.0 / (1.0 + self.dr * epoch))
        return G_loss, D_loss, files

model = numpyGAN(label_subset=[0])
gloss,dloss,files = model.train()

    

