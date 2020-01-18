# Face Generator GAN
## Creator: Jay Mody
Project from Udacity's Nanodegree program, helper.py is not mine

This project is a deep convolutional generative adveserial network. One instance of the network was trained on the mnist dataset and another on the celeba dataset.


## Model Summary
The model is a simple convolutional GAN, illustrated by the below figure from https://skymind.ai/wiki/generative-adversarial-network-gan. I used leaky relu activation functions for the hidden layers. The output of the generator was put through a sigmoid function, the output of the discriminator was put through the hyperbolic tanget function.

![GAN](/imgs/figs/gan_architecture.png)



**Generator Summary**
1. 128 vector of random uniform noise as input
2. Convolutional transpose to a 7 by 7 by 256
3. Convolutional tranpose to a 14 by 14 by 128
4. Convolutional tranpose to final output shape of 28 by 28 by (# of colour channels)

**Discriminator Summary**
1. 28 by 28 by (#colour channels) image as input
2. Convolution (filters = 64, kernel = 5, strides = 2, padding = same)
3. Convolution (filters = 128, kernel = 5, strides = 2, padding = same)
4. Convolution (filters = 256, kernel = 5, strides = 2, padding = same)
5. Flatten
5. Dense layer with 1 output node



## MNIST GAN
This was mainly to test the gan architecutre before putting it through the celeba dataset since it is much quicker to train than the celeba dataset.
Here is an example of some of the images from the mnist database:

![Mnist Examples](/imgs/mnist_examples.PNG)


Here were the final results.:

![Output Mnist](/imgs/mnist1546056472.8402486.png)


As you can see, the GAN suffers from the mode collapse problem. The numbers 3, 8, and 9 take over as the dominant outputs for the generator. Most of the numbers are also poorly generated because of the same problem, as the network cant decide whether to generate a 0 or an 8 and ends up making a mushy mess of a number, an average of the two.

Here's a plot of the loss. The generator is attempting to maximize loss, while the discriminator is attempting to minimize loss. This goes on until equilibrium is reached.

![Loss Mnist](/imgs/figs/mnist_loss.png)



## Celeba GAN
The goal of this network was to generate new images of human faces using the celeba dataset and the GAN structure tested on the mnist dataset. This network was trained only over one epoch since the dataset is quite large. Training took about 5-10 minutes, at which point it was stopped due to a lack of change in the loss. 

Here are a few examples from the celeba dataset:

![Celeba Examples](/imgs/celeba_examples.PNG)

Here are the images over the training process:

![Output Celeba](/imgs/celeba1546056556.6668892.png)
![Output Celeba](/imgs/celeba1546056576.9596279.png)
![Output Celeba](/imgs/celeba1546056596.6655672.png)
![Output Celeba](/imgs/celeba1546056615.1315722.png)
![Output Celeba](/imgs/celeba1546056633.2137961.png)
![Output Celeba](/imgs/celeba1546056651.3778517.png)
![Output Celeba](/imgs/celeba1546056669.2986689.png)
![Output Celeba](/imgs/celeba1546056687.8037908.png)
![Output Celeba](/imgs/celeba1546056706.8643024.png)
![Output Celeba](/imgs/celeba1546056725.2530332.png)
![Output Celeba](/imgs/celeba1546056744.510894.png)
![Output Celeba](/imgs/celeba1546056763.5018742.png)
![Output Celeba](/imgs/celeba1546056782.267733.png)
![Output Celeba](/imgs/celeba1546056800.4599693.png)
![Output Celeba](/imgs/celeba1546056819.2577803.png)
![Output Celeba](/imgs/celeba1546056837.559078.png)
![Output Celeba](/imgs/celeba1546056856.6104996.png)
![Output Celeba](/imgs/celeba1546056875.2741287.png)
![Output Celeba](/imgs/celeba1546056894.0428317.png)
![Output Celeba](/imgs/celeba1546056911.9931452.png)

**Final Output Result:**

![Output Celeba](/imgs/celeba1546062217.4414172.png)

The final output as you can see is identifiable as human faces. Some of the outputs didnt turn out so well however. You can see how at the very beggining, the generator pretty much spat out the random noise it was given, but by the end it learned to map that noise to generate new human faces. Intrestingly, some of the outputs in the middle are better than the final outputs, and when you look at the loss of the discriminator something intresting happened:

![Loss Celeba](/imgs/figs/celeba_loss.png)


I'm not completely sure why this spike in loss occured in the middle of the training process, especially since the discriminator is suppose to minimize loss. The generator also has this wierd affect during the same period, however to a lesser degree. It's intresting because the losses were seemingly converging without any sign of stopping, until the middle where it started fluctuating, which makes me wonder if I trained for longer (even though the loss was not changing much) if the loss for either would change like that again and produce a different, possibly more accurate result.