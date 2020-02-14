# Generative Adversarial Networks in Numpy & PyTorch

A *work in progress* implementing variations of generative adversarial networks to improve my theoretical understanding and to gain exposure with Pytorch software. Open to suggestions and feedback.

![LeCun](https://www.import.io/wp-content/uploads/2017/06/Import.io_quote-image4-170525.jpg)

## Table of Contents: 
* [GAN-numpy](#GAN-numpy)
* [GAN](#GAN)
* [CGAN](#CGAN)

## GAN-numpy







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

