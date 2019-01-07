### How to use

1.change the parameters below in script

```cpp
caffe_root: you build path of pycaffe
DEPLOY_FILE: prototxt path of original model
MODEL_FILE: caffemodel path of original model

svd_layer_info: dict of convolutions you want to prune, format like this
{layer name: {rank: rank number, input_num: input channel number of this layer}, ...}
new_prototxt_filename: the pruned model
prine_net_save_name: prune_net_save_name

```

2.run it 
```angular2html
python2 svd_prune.py
```
(default is an example that prune the openpose model used to predict the human joints position)

### Contributors: 

Tiantian Yan, Gorden Ren, Shikun Feng, and thanks to Shiwei Hou who provided the paper

```angular2html
@article{
title={DAC: Data-free Automatic Acceleration of Convolutional Networks},
author={Xin Li, Shuai Zhang, Bolan Jiang, Yingyong Qi, Mooi Choo Chuah, Ning Bi},
journal={arXiv:1812.08374},
year={2018}
}
```
