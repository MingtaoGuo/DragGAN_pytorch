# DragGAN_stylegan3
Unofficial implementation of DragGAN with StyleGAN2/3 pretrained models

## Description   
--------------

This repo is mainly to re-implement DragGAN based on stylegan2/3
- DragGAN: [Drag Your GAN: Interactive Point-based Manipulation on the
Generative Image Manifold](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)

![](https://github.com/MingtaoGuo/DragGAN_stylegan3/blob/main/resources/intro.png)
## Web Demo 
Try out the Web Demo for dragging your own image: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wuutiing2/DragGAN_pytorch)
## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN
- Python 3

### Installation
- Clone the repository:
``` 
git clone https://github.com/MingtaoGuo/DragGAN_stylegan3.git
cd DragGAN_stylegan3
```
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/) or Docker. 
All dependencies for defining the environment are provided in `environment.yaml` and `Dockerfile` .

### Dragging
Downloading the stylegan2 pretrained models:
- [stylegan2-ffhq-512x512.pt](https://drive.google.com/file/d/1OFbkHKkBOtrskyDbOrgVfjoq2eKIiTMs/view?usp=share_link)
- [stylegan2-afhqwild-512x512.pt](https://drive.google.com/file/d/1L4YN1iVC8urhW6EqCzJzJAa_Gz0_Ik7M/view?usp=share_link)
- [stylegan2-afhqdog-512x512.pt](https://drive.google.com/file/d/1pRqs6AEHaAkPaz-YbgXMSCmswhGlKnrt/view?usp=share_link)
- [stylegan2-afhqcat-512x512.pt](https://drive.google.com/file/d/1QUE-70ccfaJaYh890x-16lueoXc9y6V4/view?usp=share_link)
- [stylegan2-human-1024x1024.pt](https://drive.google.com/file/d/1cK6KHH19iF09iMUg-iUDrl_TMIq8HGhC/view?usp=share_link)

Drag generated image:
``` 
python draggan_stylegan2.py
```
Drag generated human image:
``` 
python draggan_stylegan2_human.py
```
Drag real image:
``` 
python draggan_stylegan2_realimg.py
```
In the `draggan_stylegan2.py`, `src_points (red point in image)` will be dragged to the `tar_points (blue point in image)`, so just revise the points in `src_points` and `tar_points`.
# Results
## Drag generated image
|FFHQ1|FFHQ2|
|-|-|
|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/ffhq_400.gif)|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/ffhq_600.gif)|

|Human1|Human2|
|-|-|
|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/human_100.gif)|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/human_600.gif)|

|AFHQ1|AFHQ2|
|-|-|
|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/afhq_100.gif)|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/afhq_400.gif)|

|AFHQ_Cat1|AFHQ_Cat2|
|-|-|
|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/afhq_cat_200.gif)|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/afhq_cat_800.gif)|

|AFHQ_Dog1|AFHQ_Dog2|
|-|-|
|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/afhq_dog_200.gif)|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/afhq_dog_800.gif)|
## Drag real image 
|Real image|Projected image|Drag Result|
|-|-|-|
|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/lion.png)|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/lion_proj.png)|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/afhq_lion_600.gif)|
|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/ffhq_1.png)|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/ffhq_1_proj.png)|![](https://github.com/MingtaoGuo/DragGAN_pytorch/blob/main/resources/ffhq_1.gif)|

## Author 
Mingtao Guo
E-mail: gmt798714378 at hotmail dot com
## Acknowledgement
[stylegan3](https://github.com/NVlabs/stylegan3)
[stylegan-human](https://github.com/stylegan-human/StyleGAN-Human)
[cutout team](https://www.cutout.pro/)
## Reference
[1]. Pan, Xingang, et al. "Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold." arXiv preprint arXiv:2305.10973 (2023).
