
# Multi-view Object Retrieval

Recognizing 3D objects is a complex task, particularly when dealing with texture-less objects that are only distinguishable by their shapes from certain viewpoints. Although 2D multi-view object classification achieves high accuracy when object textures are distinctive, existing methods are mostly based on supervised learning, which requires a large number of labeled images per object that are difficult to collect.

To address this challenge, we proposed a multi-loss view invariant stochastic prototype embedding method. This approach uses a progressive multi-view learning approach to minimize errors and improve recognition accuracy for novel objects from various viewpoints. This method has promising potential to improve the accuracy of recognizing texture-less objects and reduce the reliance on large amounts of labeled data for training.






## Run the code

Install Anaconda or Miniconda from Anaconda.org

To create the environment, 

```bash
  conda env create -f environment.yml -n MCB
```
Activate the environment
```bash
  conda activate MCB
```
To preprocess the data and create pickles, run the following command
```bash
  python preprocess.py
```
To run the code from scratch, run the following command
```bash
  python main.py
```
To use the pretrained model, run the following command
```bash
  python main.py --load_pretrain
```
To use different architectures, change the low_dim value to match the architecture
```bash
  #for VGG16 
  low_dim = 4096
  #for MobileNetV2
  low_dim = 1280
  #for EfficientNet B0
  low_dim = 1280
```
There may be an error like
```bash
  /path/to/.DS_Store/train folder does not exist
```
In that case, just remove the .DS_Store folder
```bash
 rm -rf /path/to/.DS_Store/