# WhiteCRC
通道剪枝
## Requirements

* python3.7.4, pytorch 1.5.1, torchvision 0.4.2, thop 0.0.31

## Reproduce the Experiment Results 

Run the following scripts to reproduce the results reported in paper (change your data path in the corresponding scripts).

* VGGNet-16-CIFAR10 ./scripts/vgg.sh
* ResNet-56-CIFAR10 ./scripts/resnet56.sh   
* ResNet-110-CIFAR10 ./scripts/resnet110.sh 
* MobileNet-v2-CIFAR10 ./scripts/mobilenetv2.sh  
* ResNet-50-ImageNet(FLOPs:2.22B) ./scripts/resnet50-1.sh  
* ResNet-50-ImageNet(FLOPs:1.50B) ./scripts/resnet50-2.sh  

## Evaluate Our Pruned Models

Run the following scripts to test our results reported in the paper (change your data path and input the pruned model path in the corresponding scripts. The pruned model can be downloaded from the links in the following table).

* VGGNet-16-CIFAR10 ./scripts/test-vgg.sh
* ResNet-56-CIFAR10 ./scripts/test-resnet56.sh   
* ResNet-110-CIFAR10 ./scripts/test-resnet110.sh 
* MobileNet-v2-CIFAR10 ./scripts/test-mobilenetv2.sh  
* ResNet-50-ImageNet(FLOPs:2.22B) ./scripts/test-resnet50-1.sh  
* ResNet-50-ImageNet(FLOPs:1.50B) ./scripts/test-resnet50-2.sh  
