## **EE541 Deep Learning - Final Project**

<p align='middle'>May 10, 2022</p>

<p align='middle'><font size='5'><b> Super Resolution</b></font></p>

<p align='middle'><font size='3'>@ YuanJi Qiu, ChengYu Zhang</font></p>



## Abstract



<p align='justify'><i><font size='4.5'>The processing to upscaling an image is called Super Resolution. In this project, we plan to use convolution neural work that can learn how to hallucinate the detailed information and collect from a large set of images. Such process does not violate the Data Processing Inequality that the information signal cannot be increased via a local physical operation. It is theoretically achievable. We propose to use CNN(Convolution Neural Network) that will input the low resolution image and train it to produce the high resolution image that matched with the original one the best. The reason we use CNN is it can be substantially deeper, more accurate, and more efficient to train if they contain shorter connections between layers close to the input and those close to the output.</font></i></p>





## Contents

[toc]





## 1. Introduction



#### 1.1 Motivation

Imagine you had a lot of photos taken in the past ten years, however it is relatively low resolution or blurred that lacks details by some facts, what if it can be restored by running certain computation and estimation the original one? This technique is called Super Resolution. By feeding the poor quality image to a computer program, which hallucinates all the details onto it, creating a crisp, high resolution image. 



#### 1.2 Techniques Survey

We found there exists lots of methods to solve super resolution with different types of model. The most traditional recipe is to pre-upsampling an image and then use normal CNN to reconstruct the details. The most popular model is SRCNN and VDSR. Another similar approach is called post-upsampling super resolution in order to reduce computation. The most representitive model is FSRCNN and ESPCN. We briefly summarize some existing model architectures.



| Model  | Full Name                                             |
| :----- | ----------------------------------------------------- |
| SRCNN  | Super  Resolution Convolution Network                 |
| VDSR   | Very Deep Super Resolution                            |
| FSRCNN | Fast Super  Resolution Convolution Network            |
| ESPCN  | Efficient Sub-Pixel Convolution Network               |
| EDSR   | Enhanced Deep Residue Network                         |
| MDSR   | Multi-scaled EDSR                                     |
| DRRN   | Deep Recursive Residual Network                       |
| SRGAN  | Super Resolution Generative and Discriminator Network |
|        |                                                       |

[BUG]



## 2. Data Preparation

#### 2.1 Datasource

We chose the DIV2k dataset. This dataset is divided into train data, validation data and test data. It has a very large diversity of contents starting from 800 high definition high resolution images we obtain corresponding low resolution images and provide both high and low resolution images for 2, 3, and 4 downscaling factors.

- **Training Dataset:**

  Selected 800 images down-sampling by 2 processed by an known downgrading operator.

- **Test Dataset:**

  Considered 100 random images as our test dataset.

|                | Resolution       | Count | Link                                                         |
| -------------- | ---------------- | ----- | ------------------------------------------------------------ |
| Training Set X | 1020$\times$768  | 800   | http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip |
| Validation X   | 1020$\times$768  | 100   | (Included above)                                             |
| Test X         | 1020$\times$768  | 100   | http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip |
| Training Y     | 2040$\times$1368 | 700   | http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip   |
| Validation Y   | 2040$\times$1368 | 100   | (Included above)                                             |
| Test Y         | 2040$\times$1368 | 100   | http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip   |



We set the proportion of the data set as 7:1:1 correspond to the training set, valid set and test set.



#### 2.2 Dataset

##### 2.1 Custom Dataset

The first step is to create a custom dataset and data loader of training. We use opencv to extract the raw RGB image to the  tensor structure. One advantage of CNN deep learning model is there is no need to crop or resize the image since the parameters we train is the kernel filter and it is not fully connected to the input data.



##### 2.2 Random Cropped Improvement

Train a 800 images with 2k resolution is very time consuming. Therefore we cropped the image into a specific size by choosing the random area from the image and collect them into a single dataset. It train much faster and the result was not influenced. Normally, we train the model at 3 stages. In the first stage, we crop the image into 128x128 randomly picked from image set and set the batch size to 10. In the second stage, we crop the image into 256x256 and 512x512 and then follow the same procedure.

<img src="/Users/randle_h/Library/Application Support/typora-user-images/image-20220510174826389.png" alt="image-20220510174826389" style="zoom:100%;" />

[BUG]



## 3. Model Evaluation

#### 3.1 Mean Absolute Error 


$$
\text{C} = \frac{\sum_{i}|{y_i-x_i}|}{N}
$$

> Pros:
>
> No matter what kind of input value, there is a stable gradient, which will not lead to the problem of gradient explosion, and has a more robust solution



>Cons:
>
>The center point is a turning point, which cannot be derived, which is inconvenient to solve, and due to the stable gradient, continuous oscillation will occur near the solution.





#### 3.2 Mean Square Error


$$
\text{C} = \frac{\sum_i(y_i-x_i)^2}{N} \\
$$

>Pros:
>
>All points are continuous and smooth, which is convenient for derivation, and has a relatively stable solution



> Cons:
>
> It is not particularly robust because when the input value of the function is far from the center value, the gradient is very large when using the gradient descent method to solve the problem, which may cause the gradient to explode (imagine that the curve continues to go up); when it is close to the true value, the curve is It becomes very smooth, and there may even be a situation where the gradient tends to 0, which causes the training speed to slow down.





#### 3.3 PSNR

**Peak Signal-to-Noise Ratio (PSNR)** is an expression for the ratio between the maximum possible value (power) of a signal and the power of distorting noise that affects the quality of its representation. Image enhancement or improving the visual quality of a digital image can be subjective.  Saying that one method provides a better quality image could vary from person to person.  For this reason, it is necessary to establish quantitative/empirical measures to compare the effects of image enhancement algorithms on image quality.
$$
\text{C} = 20\log_{10}\bigg( \frac{\text{MAX}_{I}}{\sqrt{\text{MSE}}} \bigg)
$$

|   PSNR    | Intuition                                      |
| :-------: | ---------------------------------------------- |
|   <10dB   | Very hard to identify two images               |
| 20dB~30dB | Can detect some principle structures in images |
| 30dB~45dB | Can notice some difference for human eyes      |
|   >50dB   | Hard to detect without further calculation     |
|           |                                                |



#### 3.4 SSIM

**Structure Similarity (SSIM)** is a perception-based model that considers image degradation as *perceived change in structural information*, while also incorporating important perceptual phenomena, including both luminance masking and contrast masking terms.[8]
$$
\text{SSIM}(\boldsymbol x,\boldsymbol y) =  \frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_2)(\sigma_x^2+\sigma_y^2+C_2)} \tag{1-2}
$$

> pros:
>
> Significantly and accurately evaluate the identity of two images.



> cons:
>
> Very hard and need lots of computation. Some of the algorithm is not publicly available.



## 4. Architectures



#### 4.1 SRCNN - Super Resolution Convolution Network

##### 4.1.1 Introduction

SRCNN only has three convolutional layers but first there needs to be an upsampling operation.

- **Bicubic interpolation**: Upsampled to the same size as the high-resolution (HR) image
- **Feature extraction**: Use 9×9 conv to extract image features. 
- **Feature mapping**: The 1×1 conv, it is used for non-linear mapping of the low-resolution image vector and the high-resolution image vector. 
- **Image reconstruction**: The 5×5 conv recombines these representations and reconstructs the final HR image.



Since SRCNN needs to enlarge the image to the same size as the high-resolution image outside the model, it is calculated with the HR-sized image inside the network, which will lead to a very large computational complexity:
$$
O\{(f^2_1n_1+n_1f^2_2n_2+f^2_3n_2)S_{HR} \}
$$
It is linearly proportional to the size of HR image. 



##### 4.1.2 Experiments

Our model contains three convolution layers. We pre-sampling the image using the bicubic interpolation and go through three convolution blocks.

<img src="/Users/randle_h/Library/Application Support/typora-user-images/image-20220510155536619.png" alt="image-20220510155536619" style="zoom:70%;" />

<p align='center' style="color:gray" >Figure 3: SRCNN model</p>



We impliment 2 experiments. The first one is to use a single batch which contains 800 images. The second one is to separate the dataset into 10 epochs and the batch size is 10. We cropped the image into 128x128 randomly.



##### 4.1.3 Result

The model was evaluated in term of Peak Signal-to-Noise Ratio on RGB channels. The total training process took 5 hours 14 minutes to complete. The system was implimented in Python with Pytorch and runs on GPU at Google Colab. 

<img src="/Users/randle_h/Library/Application Support/typora-user-images/image-20220510205415326.png" alt="image-20220510205415326" style="zoom:80%;" />

<p align=center style="color:gray" > Figure 4: SRCNN training curve</p> 

The PSNR was above 31 dB that indicates the output image can be well-identified. However, the actual result is still no satisfying.





#### 4.2 FSRCNN - Fast SRCNN

##### 4.2.1 Introduction

As an improved model of SRCNN, FSRCNN has a deeper depth and operates directly on LR images. The non-linear mapping in SRCNN is divided into three stages: shrinking , mapping and expanding. We use a deconvolution in the last layer to upsample and restore it to the size of the HR image.

<img src="/Users/randle_h/Library/Application Support/typora-user-images/image-20220510210045473.png" alt="image-20220510210045473" style="zoom:90%;" />

<p align=center style="color:gray" > Figure 5: Network Structures of SRCNN and FSRCNN
<a href="https://arxiv.org/pdf/1608.00367v1.pdf">Image Source</a>
</p> 



- **Feature extraction**: Bicubic interpolation and feature extraction in previous SRCNN is replaced by 5×5 conv.

- **Shrinking**: This part mainly considers that SRCNN directly performs SR on the high-dimensional feature map on the LR, so the parameters (computation amount) will be increased by increasing the number of channels and the number of filter kernels. Therefore, consider the following ideas: first reduce the number of channels of the LR image, and then operate on the low-dimensional LR feature map, which will reduce the computational complexity.

- **Feature mapping**: Multiple 3×3 layers are to replace a single wide one.

- **Expanding**: Increase the number of channels reduced by shrinking by using a 1×1 conv.

- **Deconvolution**: 9×9 filters are used to reconstruct the HR image.

   

Since FSRCNN is always calculated with the size of the LR image, only deconvolution is performed in the last step. Therefore, although it has more convolutional layers, its computational complexity is still smaller than that of SRCNN:



##### 4.2.2 Experiments

We post-upsampling the image and connect five parts to form a complete FSRCNN. On the whole, there are three sensitive variables (*i.e.,* the LR feature dimension d, the number of shrinking filters s, and the mapping depth m) governing the performance and speed[9]. In here, `m=4` `s=12` `d=56`

<img src="/Users/randle_h/Library/Application Support/typora-user-images/image-20220510165038788.png" alt="image-20220510165038788" style="zoom:80%;" />

<p align='center' style="color:gray" >Figure 6: FSRCNN model</p>

Following SRCNN, we adopt the mean square error (MSE) as the cost function.



##### 4.2.3 Result

For each epoch, we record and average the loss and psnr when validating the data.

We separate the entire training process. After running 10 epochs, we pause, save the parameter and modify the dataset by changing the crop size.

<img src="/Users/randle_h/Library/Application Support/typora-user-images/image-20220510204816264.png" alt="image-20220510204816264" style="zoom:80%;" />

<p align=center style="color:gray" > Figure 7: FSRCNN training curve</p> 



The result was not satisfied. As you can see, the PSNR is not even above 30dB that indicates the out images probably are blurred or opaque. We think the major reason is our convolution layer is not deep enough to when applying the non-linear mapping. Another possible fact is we use ReLU function instead of the PReLU which was claimed as better than ReLU.



#### 4.3 BTSRN - Balanced 2 Stage Super Resolution Network

##### 4.3.1 Introduction

Deep residual structure is necessary in both the low and high resolution stages.

Here is the model we referenced. It mainly contains two stages : (LR) low resolution stages and high resolution stages. The two stages connected with up-sampling layers.

<img src="/Users/randle_h/Library/Application Support/typora-user-images/image-20220510172618199.png" alt="image-20220510172618199" style="zoom:80%;" />

![image-20220510172700649](/Users/randle_h/Library/Application Support/typora-user-images/image-20220510172700649.png)

<p align=center style="color:gray" > Figure 8: A typical BTSRN model.
<a href="https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Fan_Balanced_Two-Stage_Residual_CVPR_2017_paper.pdf">Image Source</a>
</p> 
We found four representative residual blocks from researches.[7] It was preposed to use PConv block to achieve good trade-off between the accuracy and the speed. Compared to the others, it did not use the non-linear activation which made the model simple and efficient.

The proposed PConv blocks are employed with 128 nodes as input and 64 nodes after 1x1 convolution layer. The networks are trained with training and validation dataset, totally 900 images, and evaluated on the 100-image testing set.[7] 



##### 4.3.2 Experiments

During the training, all the images are randomly cropped into 128x128 and 256x256 patches seperately. Each step takes 10 images as a batch. The initial learning rate was set to 0.0001 and is exponentially decreased. The loss function was defined as the mean square error on RGB channels.

We noticed that the data should be copied and cloned properly in a single residual block to prevent 0 gradient error when back tracking.

<img src="/Users/randle_h/Library/Application Support/typora-user-images/image-20220510164847176.png" alt="image-20220510164847176" style="zoom:80%;" />

<p align='center'>Figure 9: BTSRN model</p>



During the test, We use PSNR method to evaluate the result. The output result was mapped from 0 to 255 and converted into `uint8` data type.







#### 4.4 Model Comparison 

[BUG]



## Appendix A

#### Problem 1

[Issue]  The output image color is reversed.

[Solution] When extracting an image using the OpenCV, we should be careful with the color depth order. The default one is `BGR`. Therefore, we should add the enumerable parameter `cv2.COLOR_BGR2RGB` to fix.



#### Problem 2

[Issue] Images inside a single batch is not equal length.

[Solution] Crop the image into equal length.



#### Problem 3

[Issue] When recording the loss and psnr values, the memory got crashed oftenly.

[Solution] When calculating the loss, use `xxxx.detach().numpy()` instead so that the data will not be count in the graph.



## Appendix B

#### Source Code Manual Reference

##### class PNGDataset

| Method / Function                | Description                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| `__init__(self, lr_dir, hr_dir)` | Given two directory that is the image path, extract the list of all images |
| __`getitem__(self, idx)`         | Given the index, return its corresponding image as the form of tensor |
| __`len__(self)`                  | Return the length of the dataset                             |
|                                  |                                                              |

##### class CropDataset(PNGDataset)

| Method / Function                            | Description                                                  |
| -------------------------------------------- | ------------------------------------------------------------ |
| `__init__(self, lr_dir, hr_dir, dim, scale)` | Given two directory and the size need to crop. `dim` should be in tuple form |
| __`getitem__(self, idx)`                     | Given the index, return its corresponding image as the form of tensor |
| __`len__(self)`                              | Return the length of the dataset                             |
|                                              |                                                              |

##### class BASE_NN(nn.Module)

| Method / Function                   | Description                                                  |
| ----------------------------------- | ------------------------------------------------------------ |
| `__init__(self, lr=0.0005)`         | Set the initial learning rate                                |
| `info(self)`                        | Print the layer information from the state dictionary        |
| `show_filter(self, layer)`          | Visualize the parameter filter of specific layer             |
| `start(self, t, v, n_e, val=False)` | Start to train the model. Validate when necessary by setting True |
| `save_param(self, path_o, path_m)`  | Given the directory, save the parameters                     |
| `infer(self, x)`                    | Output a super resolution image                              |
|                                     |                                                              |

##### class FSRCNN(BASE_NN): 

Inherited

##### class FSRCNN(BASE_NN):

Inherited

##### class BTSRN(BASE_NN):

Inherited

## Reference

[1] Christian Ledig, Lucas Theis, Ferenc Husza ́r. "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network", pages 4684~4686,  [arXiv:1609.04802](http://arxiv.org/abs/1609.04802)

[2]  Beaudry, Normand (2012), "An intuitive proof of the data processing inequality", *Quantum Information & Computation*, **12** (5–6): 432–441, [arXiv](https://en.wikipedia.org/wiki/ArXiv_(identifier)):[1107.0740](https://arxiv.org/abs/1107.0740)

[3] Z. Wang and A. C. Bovik, "Mean squared error: Love it or leave it? A new look at Signal Fidelity Measures," in *IEEE Signal Processing Magazine*, vol. 26, no. 1, pp. 98-117, Jan. 2009, doi: 10.1109/MSP.2008.930649.

[4] C. Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 105-114, doi: 10.1109/CVPR.2017.19.

[5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *"Deep Residual Learning for Image Recognition"*, 10 Dec 2015, arXiv:1512.03385v1

[6] Fan Y, Shi H, Yu J, et al. Balanced two-stage residual networks for image super-resolution[C]//Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017: 161-168.

[7] Anwar S, Khan S, Barnes N. A deep journey into super-resolution: A survey[J]. ACM Computing Surveys (CSUR), 2020, 53(3): 1-34.

[8] G. Chen, C. Yang and S. Xie, "Gradient-Based Structural Similarity for Image Quality Assessment," *2006 International Conference on Image Processing*, 2006, pp. 2929-2932, doi: 10.1109/ICIP.2006.313132.

[9] Dong C, Loy C C, Tang X. Accelerating the super-resolution convolutional neural network[C]//European conference on computer vision. Springer, Cham, 2016: 391-407.

[BUG]
