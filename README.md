# Generative Adversarial Networks in Numpy & PyTorch

## Objective:
*"Generative Adversarial Networks is the most intresting idea in the last ten years in machine learning."* 
                                                                      **-Yann LeCun, Director, Facebook AI**

![LeCun](https://www.import.io/wp-content/uploads/2017/06/Import.io_quote-image4-170525.jpg)



https://github.com/longenbach/DS504_GAN_HW_Submission



## Overview:
* [GAN](#GAN)
* [CGAN](#CGAN)




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

