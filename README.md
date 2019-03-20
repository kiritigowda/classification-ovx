[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/kiritigowda/MIVisionX-Classifier.svg?branch=master)](https://travis-ci.org/kiritigowda/MIVisionX-Classifier)

# Live Image Classification

This application runs know CNN image classifiers on live/pre-recorded video stream.

## MIVisionX Image Classification Control
[![MIVisionX Image Classification](MIVisionX-ImageClassification.png)](https://github.com/kiritigowda/MIVisionX-setup) 

## MIVisionX Image Classification
[![MIVisionX Image Classifier](classifier.png)](https://github.com/kiritigowda/MIVisionX-setup) 

## Usage
### Setup
Setup your system by running setup and build scripts from [MIVisionX](https://github.com/kiritigowda/MIVisionX-setup) setup project
### Build
````
git clone https://github.com/kiritigowda/MIVisionX-Classifier
cd MIVisionX-Classifier
cmake .
make
````
### Run
```
Usage: ./classifier 
        --inception <inceptionV4 weights.bin> [optional]
        --resnet50 <resnet50 weights.bin>     [optional]
        --vgg16 <vgg16 weights.bin>           [optional]
        --googlenet <googlenet weights.bin>   [optional]
        --resnet101 <resnet101 weights.bin>   [optional]
        --resnet152 <resnet152 weights.bin>   [optional]
        --vgg19 <vgg19 weights.bin>           [optional]
        --label <label text>                  [required]
        --video <video file>/<--capture 0>    [required]
```

#### weights.bin
Download or train your own caffemodel and run the [model_compiler](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules/tree/develop/utils/model_compiler) to get the corresponding weights.bin file.

1. Download or train your own caffemodel.

Here is the sample download link that contains all the prototxt: 

https://github.com/SnailTyan/caffe-model-zoo

2. Using model compiler

To convert a caffemodel into AMD NNIR model:
```
% python caffe2nnir.py <net.caffeModel> <nnirOutputFolder> --input-dims n,c,h,w [--verbose 0|1]
```

To convert an AMD NNIR model into OpenVX C code:

````
% python nnir2openvx.py <nnirInputFolder> <outputFolder>
````
The weights file will be generated and you can use that as an input for this project.

#### --label text

The labels.txt file in this project.
  
#### --video file
Test the classification on your own video. Give the path to your video.
  
#### --capture 0
If you want to test with a live cam, turn on this option.

### Example
```
./classifier --inception /PATH/TO/inceptionV4/weights.bin --resnet50/PATH/TO/resnet50/weights.bin ...
...
--vgg19 /PATH/TO/vgg19/weights.bin --label labels.txt --capture 0
```
