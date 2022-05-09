# Unrolled-GAN-for-Mixed-Guassian-Points
This repository implements UNROLLED GENERATIVE ADVERSARIAL NETWORKS using Pytorch (https://arxiv.org/pdf/1611.02163.pdf).

Main codes from https://github.com/mk-minchul/unroll_gan, you may find more detail about these codes here.

Several differences for 

1. Adding 2D Grid points generating function.
2. Use different Guassian points generating functions for both 2D Ring and 2D Grid.
3. Change the style of visualization.



### Please follow these steps to run Unrolled GAN
```
pip install -r requirements.txt
!python main.py --config yes_higher_unroll_10 --model 0   ##for 2D Ring 
!python main.py --config yes_higher_unroll_10 --model 1   ##for 2D Grid
```




