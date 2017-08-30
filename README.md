# Caffe for Object Detection

- This fork of Caffe contains code to train Deep Learning models for the task of object detection.
- Major contribution in this fork include implementation of two layers specific to object detection.
- Layer 1 being [bbox_data_layer](https://github.com/kvmanohar22/caffe/blob/obj_detect_loss/include/caffe/layers/bbox_data_layer.hpp) which reads in images as well as their corresponding annotations.
- Layer 2 being [squeezedet_loss_layer](https://github.com/kvmanohar22/caffe/blob/obj_detect_loss/include/caffe/layers/squeezedet_loss_layer.hpp) which implements the core of Object Detection loss function.

Loss function is the implementation of this paper: [https://arxiv.org/abs/1612.01051](https://arxiv.org/abs/1612.01051)

For more details pertaining to the loss function, refer this [blog post](https://kvmanohar22.github.io/GSoC/)

## Weights and Deploy Scripts

In the directory `proto`, you can find all the files necessary for running the models.

## Model Testing

For details on setting the variables for `test` phase in `proto/SqueezeDet_train_test.prototxt`, have a look at the protobuf message for [bbox_data_layer](https://github.com/kvmanohar22/caffe/blob/obj_detect_loss/src/caffe/proto/caffe.proto#L818).

```bash
./caffe test -model <deploy-prototxt> -weights <weights-file>
```

- For `<deploy-prototxt>`, use the `proto/SqueezeDet_train_test.prototxt`
- For `<weights-file>`, use the
`proto/SqueezeDet.caffemodel`


## Training your own models

### Dataset Preparation
  - Create the annotation file (<image_name>.txt) for each image in the training dataset
  - Each annotation should contain information related to the objects present in that image
  - Each annotation file should contain the following information for each object in the image:
    - `xmin` `ymin` `xmax` `ymax` `class-idx`
    - Here, `xmin`, `ymin` are the co-ordinates of the top left corner of the rectangle of bounding box
    - And, `xmax`, `ymax` are the co-ordinates of the bottom right corner of the rectangle of bounding box
    - `class-idx` is the class index to which the object within bounding box belongs to. (Indexing starts from `0`)
  - If an object contains more than one object, then the corresponding annotation file looks like:
    - `<xmin_1>` `<ymin_1>` `<xmax_1>` `<ymax_1>` `<class_idx_1>` `<xmin_2>` `<ymin_2>` `<xmax_2>` `<ymax_2>` `<class_idx_2>` `...`
  - Example: If image file is `2011_000090.jpg` and contains one object with co-ordinates `34` `45` `234` `215` and belongs to class `0`, the corresponding annotation file should be, `2011_000090.txt` with contents, `34 45 234 215 0`

### Implementation of Layer 'BboxData'
  - This is the layer which reads the images and corresponding annotations to train the model
  - Important variables to be set: [source](https://github.com/kvmanohar22/caffe/blob/obj_detect_loss/src/caffe/proto/caffe.proto#L825) and [root_folder](https://github.com/kvmanohar22/caffe/blob/obj_detect_loss/src/caffe/proto/caffe.proto#L851).
  - Each line in the `source` file should contain 2 values, `<path-to-image> <path-to-annotation>`
  - You can either provide absolute path or relative path in the `source` file.
  - If providing relative path, set the variable `root_folder` accordingly
  - ```bash
    root_folder/
               - Images/
                      - image_0.jpg
                      - image_1.jpg
                      - image_2.jpg
                      - ...
               - Annotations/
                      - image_0.txt
                      - image_1.txt
                      - image_2.txt
                      - ...
    ```
  - Example:
    ```
    layer {
       name: "data"
       type: "BboxData"
       top: "data"
       top: "bbox"
       bbox_data_param {
         source: '/home/user123/source.txt'
         batch_size: 2
         is_color: true
         shuffle: true
       }
       include {
         phase: TRAIN
       }
    }

    ```
  - This layer outputs two blobs, blob `data` which contains image pixel values and the blob `bbox` which contains the bounding box information
