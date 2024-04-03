# GAN_Dualism
GAN with Dualism structure for verifying


environment
Tensor flow 1.15

Code:

1 : Dualism structure for futures
  DualismTrain.py # sample code for Dualism GAN for futures Strategy. To execute, simply place the data and program in the same folder.

  PreDataTest1.csv / PreDataTest1.csv / PreDataTrain.csv / PricDataTrain.csv # The futures data DualismTrain.py requires (collected and organized by myself) to operate.
  Data in website : https://drive.google.com/drive/folders/10rw2nIxtssIeQi24DGPHOU3kl74SoLrN?usp=drive_link

  Dualism has two states. When no input data as x, it keeps in balance. left & right keep original state。
  When input the real data as x, left & right are similar to x and keep in balance

  a : original state : Not input data as x. Input data is same as output as picture state 1. 

  b : data input state : Input the real data as x. As picture state 2.
  At this point, the overall distance is determined by the difference between left and x, and the position of right in space depends on the difference between right and left, forming a contrast between right and left

![Dualism_s1](https://github.com/Alvin4862/GAN_Dualism/assets/32213728/b17a07d5-e784-4267-90a9-d50e3f7b513a)
![Dualism_s2](https://github.com/Alvin4862/GAN_Dualism/assets/32213728/9a8c7f04-9c7a-43e8-9522-7262340f6d78)

  The Dualism GAN. The structure I want is shown below.
  x	: real data 
  left 	: left generator output
  right 	: right generator output
  
  state 1 : oringinal
  No data input. For Dualism, it can keep in balance.
  
  left discriminator :
  low:right / mid:left / high:right
  right discriminator :
  low:left / mid:right / high:left
  
  state 2 : data input
  Input the real data as x.
  In Dualism, it can maintain balance, with left and right gradually becoming similar to real data and then achieving equilibrium.
  
  left discriminator
  low:right / mid:left / high:x
  right discriminator
  low:left / mid:right / high:x

2:(This does not conform to the Dualism framework because there is no opposition / MNIST database.)
  DualGenerator.py # sample code for Dual Generator. To execute directly
