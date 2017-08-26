# Caffe for Object Detection

- This fork of Caffe contains code to train Deep Learning models for the task of object detection.
- Major contribution in this fork include implementation of two layers specific to object detection.
- Layer 1 being [bbox_data_layer](https://github.com/kvmanohar22/caffe/blob/obj_detect_loss/include/caffe/layers/bbox_data_layer.hpp) which reads in images as well as their corresponding annotations.
- Layer 2 being [squeezedet_loss_layer](https://github.com/kvmanohar22/caffe/blob/obj_detect_loss/include/caffe/layers/squeezedet_loss_layer.hpp) which implements the core of Object Detection loss function.

Loss function is the implementation of this paper: [https://arxiv.org/abs/1612.01051](https://arxiv.org/abs/1612.01051)

For more details pertaining to the loss function, refer this [blog post](https://kvmanohar22.github.io/GSoC/)
