import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from numpy import linalg as la

# Make sure that caffe is on the python path:
caffe_root = '/media/kun/1a336f66-00a2-44bf-8632-f0901e0efe2a/kun/Documents/pose_estimate/caffe_rtpose_mk'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2


import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
from caffe import layers as L


# load model
DEPLOY_FILE = '/media/kun/1a336f66-00a2-44bf-8632-f0901e0efe2a/kun/Documents/pose_estimate/caffe_rtpose/model/coco/pose_deploy_linevec.prototxt'
MODEL_FILE = '/media/kun/1a336f66-00a2-44bf-8632-f0901e0efe2a/kun/Documents/pose_estimate/caffe_rtpose/model/coco/pose_iter_440000.caffemodel'
# run the net and return prediction
net = caffe.Net(DEPLOY_FILE, MODEL_FILE, caffe.TEST)


def Decompose(M, SVD_R, Kw, Kh, N):
    # print ("SVD %d" % SVD_R)
    u, sigma, vt = la.svd(M)
    # print ("Sigma: ", sigma)
    # print ("len(sigma): ", len(sigma))
    if SVD_R > len(sigma):
        print ("SVD_R is too large :-(")
        sys.exit()
    Ur = np.matrix(u[:, :SVD_R])
    # Sr = sigma[:SVD_R, :SVD_R]
    Sr = np.matrix(np.diag(sigma[:SVD_R]))
    Vr = vt[:SVD_R, :]
    # print ("Ur", Ur.shape)
    # print ("Sr", Sr.shape)
    # print ("Vr", Vr.shape)

    D = Vr.reshape(SVD_R, Kw, Kh, 1)
    S = Ur * Sr
    S = np.array(S[:])
    # print S.shape
    S = S.reshape(N, 1, 1, SVD_R)
    # print S.shape
    return D,S

def NetProcess(prob, SVD_R):
    N = prob.shape[0]
    C = prob.shape[1]
    Kw = prob.shape[2]
    Kh = prob.shape[3]
    list_d = np.zeros((C, SVD_R, Kw, Kh, 1), dtype=np.float32)
    list_s = np.zeros((N, 1, 1, C, SVD_R), dtype=np.float32)
    for i in range(C):
        Mi = prob[:,i,:,:].reshape(N, Kw * Kh)
        Di, Si = Decompose(Mi, SVD_R, Kw, Kh, N)
        # print "Di: ",Di.shape
        # print "Si: ",Si.shape
        list_d[i, :, :, :, :] = Di
        list_s[:, :, :, i, :] = Si
    Td = list_d.reshape(SVD_R * C, Kw, Kh, 1)
    Ts = list_s.reshape(N, 1, 1, SVD_R * C)
    return Td,Ts



def Decompose2(M, SVD_R, Kw, Kh, N):
    # print ("SVD %d" % SVD_R)
    u, sigma, vt = la.svd(M)
    #SVD_R = len(sigma)
    #print(SVD_R)
    # print ("Sigma: ", sigma)
    # print ("len(sigma): ", len(sigma))
    if SVD_R > len(sigma):
        print ("SVD_R is too large :-(")
        sys.exit()
    Ur = np.matrix(u[:, :SVD_R])
    # Sr = sigma[:SVD_R, :SVD_R]
    Sr = np.matrix(np.diag(sigma[:SVD_R]))
    Vr = vt[:SVD_R, :]
    # print ("Ur", Ur.shape)
    # print ("Sr", Sr.shape)
    # print ("Vr", Vr.shape)

    D = Vr.reshape(SVD_R, 1, Kw, Kh)
    S = Ur * Sr
    S = np.array(S[:])
    # print S.shape
    S = S.reshape(N, SVD_R, 1, 1)
    # print S.shape
    return D,S

def NetProcess2(prob, SVD_R):
    N = prob.shape[0]
    C = prob.shape[1]
    Kw = prob.shape[2]
    Kh = prob.shape[3]
    list_d = np.zeros((C, SVD_R, 1, Kw, Kh), dtype=np.float32)
    list_s = np.zeros((N, C, SVD_R, 1, 1), dtype=np.float32)
    for i in range(C):
        Mi = prob[:,i,:,:].reshape(N, Kw * Kh)
        Di, Si = Decompose2(Mi, SVD_R, Kw, Kh, N)
        # print "Di: ",Di.shape
        # print "Si: ",Si.shape
        list_d[i, :, :, :, :] = Di
        list_s[:, i, :, :, :] = Si
    Td = list_d.reshape(SVD_R * C, 1, Kw, Kh)
    Ts = list_s.reshape(N, C*SVD_R, 1, 1)
    return Td,Ts



# name : rank number
# svd_layer_info = {'conv5_1_CPM_L1': {"rank": 5, "input_num": 128}}

svd_layer_info = {'Mconv5_stage6_L1': {"rank": 20, "input_num": 128}, 'Mconv4_stage6_L1': {"rank": 20, "input_num": 128}, 'Mconv3_stage6_L1': {"rank": 20, "input_num": 128}, 'Mconv2_stage6_L1': {"rank": 20, "input_num": 128}, 'Mconv1_stage6_L1': {"rank": 20, "input_num": 185},
                  'Mconv5_stage6_L2': {"rank": 20, "input_num": 128},
                  'Mconv4_stage6_L2': {"rank": 20, "input_num": 128},
                  'Mconv3_stage6_L2': {"rank": 20, "input_num": 128},
                  'Mconv2_stage6_L2': {"rank": 20, "input_num": 128}, 'Mconv1_stage6_L2': {"rank": 20, "input_num": 185}}


# svd_layer_info = {'conv5_1_CPM_L1': {"rank": 5, "input_num": 128}, 'conv5_2_CPM_L1': {"rank": 5, "input_num": 128}, 'conv5_3_CPM_L1': {"rank": 5, "input_num": 128},
#                   'conv5_1_CPM_L2': {"rank": 5, "input_num": 128},
#                   'conv5_2_CPM_L2': {"rank": 5, "input_num": 128},
#                   'conv5_3_CPM_L2': {"rank": 5, "input_num": 128}}
#
# # svd_layer_info = {'Mconv1_stage6_L1': {"rank": 20, "input_num": 185}, 'Mconv1_stage6_L2': {"rank": 20, "input_num": 185}}
#
# svd_layer_info_meta = {'Mconv5_stage5_L1': {"rank": 20, "input_num": 128}, 'Mconv4_stage5_L1': {"rank": 20, "input_num": 128}, 'Mconv3_stage5_L1': {"rank": 20, "input_num": 128}, 'Mconv2_stage5_L1': {"rank": 20, "input_num": 128}, 'Mconv1_stage5_L1': {"rank": 20, "input_num": 185},
#                   'Mconv5_stage5_L2': {"rank": 20, "input_num": 128},
#                   'Mconv4_stage5_L2': {"rank": 20, "input_num": 128},
#                   'Mconv3_stage5_L2': {"rank": 20, "input_num": 128},
#                   'Mconv2_stage5_L2': {"rank": 20, "input_num": 128}, 'Mconv1_stage5_L2': {"rank": 20, "input_num": 185}}
#
# # svd_layer_info = {}
# for i in range(1):
#     for k in svd_layer_info_meta:
#         new_k = k.replace('stage5', 'stage' + str(i + 2))
#         svd_layer_info[new_k] = svd_layer_info_meta[k]
#         svd_layer_info[new_k]['rank'] = 10

# svd_layer_info = svd_layer_info_meta
prototxt_net = caffe_pb2.NetParameter()
# prototxt_net = caffe.NetSpec()
with open(DEPLOY_FILE, 'r') as f:
    pb.text_format.Merge(f.read(), prototxt_net)
# new_layers = []

order_layer_list = []
for layer in prototxt_net.layer:
    order_layer_list.append(layer)
    if layer.type == "Convolution" and layer.name in svd_layer_info:
        svd_rank = svd_layer_info[layer.name]['rank']
        last_layer_name = layer.bottom[0] # only consider one input layer
        input_channel_number = svd_layer_info[layer.name]['input_num']

        layer.convolution_param.group = input_channel_number
        old_num_output = layer.convolution_param.num_output
        layer.convolution_param.num_output = int(svd_rank * input_channel_number)
        top_names = [ele for ele in layer.top]
        layer.top[:] = [layer.name + '_depb']
        # layer.top.append(layer.name + '_depb')

        # new a depthwise layer
        dp_layer = L.Convolution(kernel_size=1, num_output=old_num_output, pad=0)
        # layer, name = layer.name + '_depth',

        # new_conv = prototxt_net.layer.add()
        new_conv = caffe_pb2.LayerParameter()
        new_conv.CopyFrom(dp_layer.to_proto().layer[0])
        new_conv.name = layer.name + '_dep'
        new_conv.bottom.append(layer.name + '_depb')
        new_conv.top[:] = top_names
        order_layer_list.append(new_conv)

del prototxt_net.layer[:]
prototxt_net.layer.extend(order_layer_list)
# save as the new prototxt
# prototxt_net.layer.sort()
new_prototxt_filename = "/home/kun/PycharmProjects/CaffeSVDPrune/new_pose.prototxt"
with open(new_prototxt_filename, 'w') as f:
    f.write(pb.text_format.MessageToString(prototxt_net))

# Got the new prototxt, now we need the parameters


prune_net = caffe.Net(new_prototxt_filename, caffe.TEST)
prune_net_param = prune_net.params
for layer_name, param in net.params.iteritems():
    # prune_net_param[layer_name] = param
    # continue
    if layer_name not in svd_layer_info:
        for ele_new, ele_old in zip(prune_net_param[layer_name], param):
            ele_new.data[:] = ele_old.data[:]
    else:
        conv_data = param[0].data
        bias_data = param[1].data
        print bias_data
        td, ts = NetProcess2(conv_data,  svd_layer_info[layer_name]['rank'])
        # c = prune_net_param[layer_name][0].data.shape[1]
        # for i in range(c):
        #     prune_net_param[layer_name][0].data[:, i, :, :] = td[:, :, :, i]
        prune_net_param[layer_name][0].data[:] = td[:]

        # prune_net_param[layer_name][0].data[:] = td.swapaxes(3, 2).swapaxes(2, 1)
        # prune_net_param[layer_name][0].data[:] = td.reshape(prune_net_param[layer_name][0].data.shape)
        prune_net_param[layer_name][1].data[:] = np.zeros(shape=prune_net_param[layer_name][1].data.shape)
        # c = prune_net_param[layer_name + '_dep'][0].data.shape[1]
        # for i in range(c):
        #     prune_net_param[layer_name + '_dep'][0].data[:, i, :, :] = ts[:, :, :, i]
        prune_net_param[layer_name + '_dep'][0].data[:] = ts[:]

        # prune_net_param[layer_name + '_dep'][0].data[:] = ts.swapaxes(3, 2).swapaxes(2, 1)
        # prune_net_param[layer_name + '_dep'][0].data[:] = ts.reshape(prune_net_param[layer_name + '_dep'][0].data.shape)
        prune_net_param[layer_name + '_dep'][1].data[:] = bias_data

prune_net_save_name = "/home/kun/PycharmProjects/CaffeSVDPrune/pose_prune.caffemodel"
prune_net.save(prune_net_save_name)
print 'prune finished!!!'
