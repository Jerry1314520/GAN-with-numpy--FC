# Generative Adversarial Networks in Numpy & PyTorch

A *work in progress* implementing variations of generative adversarial networks to improve my theoretical understanding and to gain exposure with Pytorch software. Open to suggestions and feedback.

![LeCun](https://www.import.io/wp-content/uploads/2017/06/Import.io_quote-image4-170525.jpg)

## Table of Contents: 
* [GAN-numpy](#GAN-numpy)
* [GAN](#GAN)
* [CGAN](#CGAN)

## GAN-numpy

### Generator Architecture:
![art_G](imgs/GAN_numpy_G.jpeg)

```python 
## Intial Generator Weights:
self.G_W1 = np.random.randn(self.noise_dim, 128) * np.sqrt(2. / self.noise_dim) # 100 x 128
self.G_b1 = np.zeros((1, 128)) # 1 x 128

self.G_W2 = np.random.randn(128, 256) * np.sqrt(2. / 128) # 128 x 256
self.G_b2 = np.zeros((1, 256)) # 1 x 256

self.G_W3 = np.random.randn(256, self.img_size ** 2) * np.sqrt(2. / 256) # 256 x 784
self.G_b3 = np.zeros((1, self.img_size ** 2)) # 1 x 784
    
## Generator:
def generator_forward(self, noise):
    self.G_a1 = self.hidden_layer_forward(noise,self.G_W1,self.G_b1,acitvation='relu')
    self.G_a2 = self.hidden_layer_forward(self.G_a1,self.G_W2,self.G_b2,acitvation='relu')
    self.G_a3 = self.hidden_layer_forward(self.G_a2,self.G_W3,self.G_b3,acitvation='tanh')
    '''ADD MORE LAYERS'''
    return self.G_a3
```

### Discriminator Architecture:
![art_D](imgs/GAN_numpy_D.jpeg)

```python 
## Intial Discrimnator Weights::
self.D_W1 = np.random.randn(self.img_size ** 2, 128) * np.sqrt(2. / self.img_size ** 2) # 784 x 128
self.D_b1 = np.zeros((1, 128)) # 1 x 128

self.D_W2 = np.random.randn(128, 1) * np.sqrt(2. / 128) # 128 x 1
self.D_b2 = np.zeros((1, 1)) # 1 x 1

## Discriminator:
def discriminator_forward(self, img):
    self.D_a1 = self.hidden_layer_forward(img,self.D_W1,self.D_b1,acitvation='relu')
    self.D_a2 = self.hidden_layer_forward(self.D_a1,self.D_W2,self.D_b2,acitvation='sigmoid')
    '''ADD MORE LAYERS'''
    return self.D_a2
```


## GAN:








## CGAN:





```python
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
```

<p align="center">
  <img width="300" height="300" src=imgs/CGAN_output.gif>
</p>

![CGANloss](imgs/CGAN_loss.png)

