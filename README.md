# Pancreas CT image and volumetric segmentation


## Dataset
The pancreas computed tomography (CT) [dataset](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU) consists of 18,942 DICOM images of 80 patients, with a total size of 9.3GB, and their corresponding manually annotated masks as the ground-truths. 

## Objective 

Automatic image (2D) and volume (3D) segmentation of pancreas computed tomography (CT).

## Evaluation Metrics
F2 score and dice similarity coefficient (DSC).

## Solution Approach
For each patient, there are several medical images where each image is a single slice of the scan with a resolution of 512×512 pixels. Among patients, the number of slices varies from 181 to 466 with a median of 237. The annotation masks label anomalies and are in Neuroimaging Informatics Technology Initiative (NIfTI) format. Regions in the CT scan slices with pixel values of 1 and 0 denote areas with and without anomalies, respectively. With this dataset, I perform both 2D and 3D medical image segmentation. In 2D, I consider each slice on its own, and in 3D, I consider the volume built on the collection of slices of each patient.

I use 2D-UNet and 3D-Unet which are fully convolutional networks developed for biomedical image segmentation. The 2D UNet architecture contains an encoder and a decoder path each with four resolution steps. The encoder path captures the context of the image and the decoder path enables localization. Each of the encoder and decoder paths consists of 4 blocks which contain two 3×3 convolutions each followed by a batch normalization for faster convergence and a rectified linear unit (ReLU), and then a 2×2 max pooling with strides of two in each dimension. In the decoder path, each layer consists of an upconvolution of 2×2 by strides of two in each dimension, followed by two 3×3 convolutions each followed by a ReLU. Through shortcut connections from layers of equal resolution in the encoder path, the essential high-resolution features are provided to the decoder path. To avoid bottlenecks, the number of feature channels is doubled after each maxpooling and halved in each upconvolution. To avoid overfitting, I use dropout with a ratio of 20% after each maxpooling and upconvolution. In the last layer, a 1×1 convolution reduces the number of output channels to 1. Then, I apply a sigmoid along with each pixel to form the loss. I have initialized the weights using Gaussian distribution with a standard deviation of  √(2⁄N),, where N is the number of incoming nodes of one neuron. 
3D UNet architecture is an extension of 2D UNet, where all 2D operations are replaced by their 3D counterparts: 3D convolutions, 3D maxpooling and 3D up-convolutional layers. Figure 1 shows a sample of the 3D UNet with 32 input features for a patch of 64×128×128 voxels.

<p align="center">
<image src= "assets/3D%20UNet.png" width="600"> 
</p> 
<p align="center">               
Figure 1. 3D UNet architecture
</p>

## Results

<p align="center">
<image src= "assets/manual%20annotation.jpg" width="600"> 
</p> 
<p align="center">               
Figure 2. Manual segmentation
</p>


<p align="center">
<image src= "assets/automatic%20segmentation.jpg" width="600"> 
</p> 
<p align="center">               
Figure 3. Automatic segmentation
</p>


References:

https://github.com/fepegar/unet/tree/v0.7.5

https://github.com/albumentations-team/albumentations 

https://github.com/fepegar/torchio

https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py

https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb

https://discuss.pytorch.org/t/pytorch-how-to-initialize-weights/81511/4

https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/24

https://github.com/joe-siyuan-qiao/WeightStandardization

https://discuss.pytorch.org/t/patch-making-does-pytorch-have-anything-to-offer/33850/11

https://discuss.pytorch.org/t/fold-unfold-to-get-patches/53419

https://github.com/davidtvs/pytorch-lr-finder

https://github.com/frankkramer-lab/MIScnn

Roth HR, Lu L, Farag A, Shin H-C, Liu J, Turkbey EB, Summers RM. DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation. N. Navab et al. (Eds.): MICCAI 2015, Part I, LNCS 9349, pp. 556–564, 2015.

Cancer imaging archive wiki. (n.d.). Retrieved February 06, 2021, from https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT#225140409ddfdf8d3b134d30a5287169935068e3.

Ronneberger, O.; Fischer, P.; Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. In Proceedings of the International Conference on Medical image computing and computer-assisted intervention, Munich, Germany, 5–9 October 2015; pp. 234–241.

Çiçek, Ö.; Abdulkadir, A.; Lienkamp, S.S.; Brox, T.; Ronneberger, O. 3D U-Net: Learning dense volumetric segmentation from sparse annotation. In Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention, Athens, Greece, 17–21 October 2016; pp. 424–432.

Salehi, Seyed Sadegh Mohseni et al. “Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks.” ArXiv abs/1706.05721 (2017): n. pag.

Pérez-García, F., Sparks, R., Ourselin, S.: TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning. arXiv:2003.04696 [cs, eess, stat] (Mar 2020), http://arxiv.org/abs/2003.04696, arXiv: 2003.04696

https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6

L. N. Smith, A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay, US Naval Research Laboratory Technical Report 5510-026, arXiv:1803.09820v2. 2018

Dataset

Holger R. Roth, Amal Farag, Evrim B. Turkbey, Le Lu, Jiamin Liu, and Ronald M. Summers. (2016). Data From Pancreas-CT. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU
