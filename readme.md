### How to use
1. change the parameters below in script

```cpp
caffe_root: you build path of pycaffe
DEPLOY_FILE: prototxt path of original model
MODEL_FILE: caffemodel path of original model

svd_layer_info: dict of convolutions you want to prune, format like this
{layer name: {rank: rank number, input_num: input channel number of this layer}, ...}
new_prototxt_filename: the pruned model
prine_net_save_name: prune_net_save_name

```
2. run it (default is an example that prune the openpose model used to predict the human joints position)