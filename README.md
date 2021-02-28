# Deep Learning Projects
This repository is a work in progress and has code implementing different deep learning tasks. Install the required packages using
```
pip install -r requirements.txt
```
### Multi-class Classification
#### 1. Example using MNIST and LeNet ([LeNet.ipynb](LeNet.ipynb))
   
   Training Metrics: Loss and Accuracy
   ![](docs/train_graph_lenet.png "Training Graphs")
   
   Validation Metrics: Loss and Accuracy
   ![](docs/validation_graph_lenet.png "Validation Graphs")

   
#### 2. Notebook using ResNet to classify a 150 class [Kaggle dataset](https://www.kaggle.com/thedagger/pokemon-generation-one) ([ResNet.ipynb](ResNet.ipynb)).
  
   Also experimented with [VGG](models/VGG16.py) for the same dataset. Both models were fine-tuned from a model pretrained on ImageNet. 

   Training Metrics: Loss and Accuracy
   ![](docs/train_graph_resnet_and_vgg16.png "Training Graphs")
   
   Validation Metrics: Loss and Accuracy
   ![](docs/validation_graph_resnet_and_vgg16.png "Validation Graphs")


### Language Translation using RNNs (WIP)
### Object Detection
