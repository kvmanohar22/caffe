#ifndef CAFFE_SQUEEZEDET_LOSS_LAYER_HPP_
#define CAFFE_SQUEEZEDET_LOSS_LAYER_HPP_

#include <iostream>
#include <memory>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

/**
 *@brief Computes the regression based loss which is specific to object
 *       detection task.
 *       The details of this loss function are outlined in the paper:
 *       `https://arxiv.org/abs/1612.01051`
 *
 *@param bottom input Blob vector (length 2)
 *    -# @f$ (N \times H \times W \times C) @f$
 */

typedef struct {
  int xmin, xmax;
  int ymin, ymax;
  int indx;
} single_object;

template <typename Dtype>
class SqueezeDetLossLayer : public LossLayer<Dtype> {
 public:
  explicit SqueezeDetLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SqueezeDetLoss"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // Apply softmax activation to get class specific probability distribution
  // @f$ Pr(class_{c} \mid Object), c \in [1, C] @f$
  int batch_tot_class_probs;
  vector<int> softmax_layer_shape_;
  Dtype* softmax_layer_data_;
  shared_ptr<Layer<Dtype> > softmax_layer_;
  // Store the class specific probability predictions from the SoftmaxLayer
  Blob<Dtype> probs_;
  // bottom vector holder which is used to call SoftmaxLayer::Forward
  vector<Blob<Dtype>* > softmax_bottom_vec_;
  // top vector holder which is used to call SoftmaxLayer::Forward
  vector<Blob<Dtype>* > softmax_top_vec_;
  // values which correspond to class probability from ConvDet output
  Blob<Dtype>* softmax_input_vec_;

  // Apply sigmoid activation to get the confidence score
  // @f$ Pr(Object) @f$
  int batch_tot_conf_scores;
  vector<int> sigmoid_layer_shape_;
  Dtype* sigmoid_layer_data_;
  shared_ptr<Layer<Dtype> > sigmoid_layer_;
  // Store the confidence score from the SigmoidLayer
  Blob<Dtype> conf_;
  // bottom vector holder which is used to call SigmoidLayer::Forward
  vector<Blob<Dtype>* > sigmoid_bottom_vec_;
  // top vector holder which is used to call SigmoidLayer::Forward
  vector<Blob<Dtype>* > sigmoid_top_vec_;
  // values which correspond to confidence score from ConvDet output
  Blob<Dtype>* sigmoid_input_vec_;

  // Extract the relative bounding box coordinates
  // @f$ [\delta x_{ijk}, \delta y_{ijk}, \delta w_{ijk}, \delta h_{ijk}] @f$
  int batch_tot_rel_coord;
  vector<int> rel_coord_layer_shape_;
  Dtype* rel_coord_data_;
  // Store the relative coordinates predicted from the layer
  Blob<Dtype>* relative_coord_vec_;

  // Mode of normalization
  LossParameter_NormalizationMode normalization_;

  // Constants specific to Bounding Boxes
  int anchors_;
  int classes_;
  int pos_conf_;
  int neg_conf_;
  int lambda_bbox_;

  // Anchor shape of the form
  // @f$ [center_x, center_y, anchor_height, anchor_width] @f$
  std::vector<std::vector<std::vector<std::vector<float> > > > anchors_values_;

  // Details pertaining to the bottom blob aka output of `ConvDet layer`
  // channels, width, height
  int N, W, H, C;
  int image_height, image_width;
};

}  // namespace caffe

#endif  // CAFFE_SQUEEZEDET_LOSS_LAYER_HPP_
