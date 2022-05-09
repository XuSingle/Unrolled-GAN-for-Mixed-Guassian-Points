# Unrolled-GAN-for-Mixed-Guassian-Points
This repository implements UNROLLED GENERATIVE ADVERSARIAL NETWORKS (https://arxiv.org/pdf/1611.02163.pdf).

Main code from https://github.com/mk-minchul/unroll_gan

Several differences for 

1. Use different Guassian points generating function for both Ring and Grid.
2. Change the style of visualization.



### Please follow these steps to run Unrolled GAN
```
pip install -r requirements.txt
!python main.py --config yes_higher_unroll_10 --model 0   ##for 2D Ring 
!python main.py --config yes_higher_unroll_10 --model 1   ##for 2D Grid
```




