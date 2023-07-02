![Language](https://img.shields.io/badge/Language-Python%20-blue.svg)
![License](https://img.shields.io/badge/license-Apache_2.0-orange.svg)

# HandWave
## Abstract
A real-time sign language translator is an important milestone in facilitating communication between the deaf community and the general public. We hereby present an approach of an American Sign Language (ASL) translator based on a convolutional neural network. <br />
We are going to utilize the pre-trained SSD model architecture for the real-time recognition of the ASL using the concept of transfer learning and to give
it a try to deal with dynamic gestures. <br />
The main objectives of this project are to develop an accurate and efficient ASL recognition system, to improve communication between the deaf and hearing communities, and to promote inclusivity and accessibility for all. <br />
The system has been evaluated on a dataset of ASL gestures and has achieved an accuracy of over 90 In summary, the Real-time American Sign Language Recognition System is a promising solution for bridging the communication gap between the deaf and hearing communities. <br />
It provides an efficient, accurate, and accessible means of communicating in ASL, which can significantly improve the quality of life for individuals with hearing impairments. The system can also be extended to include additional features such as gesture recognition for other sign languages.

# Environment Setup

## IDEs and Editors
- LabelImg: An open-source Python application that can be installed to help in data preparation to add annotations to the collected images.
- Anaconda: This enables you to build different virtual environments and it’s easy to deal with.
- Visual Studio Code (VSCode): The easiest editor to develop the web application and review it using the live server extension.

## GPU Setup
Deep Learning Applications especially the model training phase here is a highly resources consumer process, it consumes time, memory,
power and CPU cycles in a very high manner, so it’s better to get the benefits of your GPU card if you have a suitable one for this project.<br />
To get your GPU card working Properly follow these steps:
1. Windows Native - Windows 7 or higher (64-bit) (no GPU support after TF 2.10).
2. Windows WSL2 - Windows 10 19044 or higher (64-bit).
3. Python 3.8 - 3.11
4. Install Visual Studio community edition: We installed VS2019.
5. Install NVIDIA GPU driver, we installed the following:
(a) CUDA Toolkit 11.2
(b) cuDNN 8.1.0
You need to sign up for the Nvidia developer program (free)
Extract all folder contents from cudnn you just downloaded to `C:/program files/Nvidia GPU computing toolkit/CUDA/v11.0`.
6. Install your IDE if you haven’t already done so, you may use `spyder`, `jupyter notebook` or `VSCode`.
7. Create a conda environment with python = 3.9 `conda create --name tf python=3.9`
8. Activate the environment in the terminal `conda activate tf`
9. Execute the following command:
`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`.
10. Install TensorFlow
```
pip install --upgrade pip
pip install tensorflow==2.5
```
11. Verify the installation with CPU
```
python -c "import tensorflow as tf;
print(tf.reduce sum(tf.random.normal([$1000, 1000$])))
```
If a tensor is returned, you’ve installed TensorFlow successfully.
12. Verify the installation with GPU
```
python -c "import tensorflow as tf;
print(tf.config.list physical devices(’GPU’))
```
If a list of GPU devices is returned, you’ve installed TensorFlow
successfully.
13. Or create a new Python file and run these lines to test if GPU
is recognized by TensorFlow.
```
import tensorflow as tf
tf.test.is gpu available(cuda only=False, min cuda compute capability=None)
```

## Data preparation
We use the LabelImg application to easily draw a box around each hand gesture in the image and give this box the suitable label of the class represented by the gesture, then split the whole data for the training and testing.<br />
- Training: 90%
- Testing: 10%



My part of the data : https://drive.google.com/file/d/137YK1DZ51LDefSdXxtnwZ70NevJB8on5/view?usp=drive_link
