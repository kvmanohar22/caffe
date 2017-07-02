#ifndef CAFFE_SQUEEZEDET_LOSS_LAYER_HPP_
#define CAFFE_SQUEEZEDET_LOSS_LAYER_HPP_

#include <iostream>  // NOLINT(readability/streams)
#include <memory>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

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
  shared_ptr<Layer<Dtype> > reshape_softmax_layer_;
  // Store the class specific probability predictions from the SoftmaxLayer
  Blob<Dtype> probs_;
  Blob<Dtype> reshape_probs_;
  // bottom vector holder which is used to call SoftmaxLayer::Forward
  vector<Blob<Dtype>* > softmax_bottom_vec_;
  vector<Blob<Dtype>* > reshape_softmax_bottom_vec_;
  // top vector holder which is used to call SoftmaxLayer::Forward
  vector<Blob<Dtype>* > softmax_top_vec_;
  vector<Blob<Dtype>* > reshape_softmax_top_vec_;
  // values which correspond to class probability from ConvDet output
  Blob<Dtype>* softmax_input_vec_;

  // Apply sigmoid activation to get the confidence score
  // @f$ Pr(Object) @f$
  int batch_tot_conf_scores;
  vector<int> sigmoid_layer_shape_;
  Dtype* sigmoid_layer_data_;
  shared_ptr<Layer<Dtype> > sigmoid_layer_;
  shared_ptr<Layer<Dtype> > reshape_sigmoid_layer_;
  // Store the confidence score from the SigmoidLayer
  Blob<Dtype> conf_;
  Blob<Dtype> reshape_conf_;
  // bottom vector holder which is used to call SigmoidLayer::Forward
  vector<Blob<Dtype>* > sigmoid_bottom_vec_;
  vector<Blob<Dtype>* > reshape_sigmoid_bottom_vec_;
  // top vector holder which is used to call SigmoidLayer::Forward
  vector<Blob<Dtype>* > sigmoid_top_vec_;
  vector<Blob<Dtype>* > reshape_sigmoid_top_vec_;
  // values which correspond to confidence score from ConvDet output
  Blob<Dtype>* sigmoid_input_vec_;

  // Extract the relative bounding box coordinates
  // @f$ [\delta x_{ijk}, \delta y_{ijk}, \delta w_{ijk}, \delta h_{ijk}] @f$
  int batch_tot_rel_coord;
  vector<int> rel_coord_layer_shape_;
  Dtype* rel_coord_data_;
  shared_ptr<Layer<Dtype> > reshape_bbox_layer_;
  // Store the relative coordinates predicted from the layer
  Blob<Dtype>* relative_coord_vec_;
  Blob<Dtype> reshape_coord_vec_;
  vector<Blob<Dtype>* > reshape_bbox_bottom_vec_;
  vector<Blob<Dtype>* > reshape_bbox_top_vec_;

  // Mode of normalization
  LossParameter_NormalizationMode normalization_;

   /**
    * @brief Constants specific to Bounding Boxes
    * - anchors_per_grid : The number of anchors at each (i, j) of ConvDet
    *                      activation map
    * - classes_ : The total number of classes of objects
    * - pos_conf_ : @f$ \lambda_{conf}^{+} @f$
    * - neg_conf_ : @f$ \lambda_{conf}^{-} @f$
    * - lambda_bbox_ : @f$ \lambda_{bbox} @f$
    * - anchors_ : The total number of anchors predicted
    *              `H * W * anchors_per_grid`
    */
  int anchors_per_grid;
  int classes_;
  int pos_conf_;
  int neg_conf_;
  int lambda_bbox_;
  int anchors_;

  // Anchor shape of the form
  // @f$ [anchors_, 4] @f$ where `4` values are as follows:
  // @f$ [center_x, center_y, anchor_height, anchor_width] @f$
  std::vector<std::vector<Dtype> > anchors_values_;

  // Anchors predictions from the `ConvDet` layer which are of shape:
  // @f$ [N, anchors_, 4] @f$ where the `4` values are as follows:
  // @f$ [delta x_{ijk}, delta y_{ijk}, delta h_{ijk}, delta w_{ijk}] @f$
  std::vector<std::vector<std::vector<Dtype> > > anchors_preds_;

  // Transformed  `ConvDet` predictions to bounding boxes which are of shape:
  // @f$ [N, anchors_, 4] @f$ where the `4` values are as follows:
  // @f$ [center_x, center_y, anchor_height, anchor_width] @f$
  std::vector<std::vector<std::vector<Dtype> > > transformed_bbox_preds_;

  // Transformed predicted bounding boxes from:
  // @f$ [N, anchors_, 4] @f$ where the `4` values are as follows:
  // @f$ [center_x, center_y, anchor_height, anchor_width] @f$ to
  // @f$ [N, anchors_, 4] @f$ where the `4` values are as follows:
  // @f$ [xmin, ymin, xmax, ymax] @f$
  std::vector<std::vector<std::vector<Dtype> > > min_max_pred_bboxs_;

  // Details pertaining to the bottom blob aka output of `ConvDet layer`
  // channels, width, height
  int N, W, H, C;
  int image_height, image_width;
  double epsilon;

  // Utility functions
  void transform_bbox(std::vector<std::vector<std::vector<Dtype> > > *pre,
    std::vector<std::vector<std::vector<Dtype> > > *post);
  void transform_bbox_inv(std::vector<std::vector<std::vector<Dtype> > > *pre,
    std::vector<std::vector<std::vector<Dtype> > > *post);

   /**
    * @brief Computes `IOU` for a batch of data.
    *
    * @param predicted_bboxs_ : [N, anchors_, 4]
    *   - N        : batch_size
    *   - anchors_ : (the total number of anchors) `H * W * anchors_per_grid`
    *   - `4`      : @f$ [xmin, ymin, xmax, ymax] @f$
    * @param min_max_gtruth_ : [N, objects_, 4]
    *   - N        : batch_size
    *   - objects_ : represents the number of objects in an image
    *   - `4`      : @f$ [xmin, ymin, xmax, ymax] @f$
    * @param iou_ : [N, objects_, anchors_]
    *   - N        : batch_size
    *   - objects_ : `IOU` value for each of the object with ground truth
    *   - anchors_ : (the total number of anchors) `H * W * anchors_per_grid`
    */
    void intersection_over_union(std::vector<std::vector<std::vector<Dtype> > >
    *predicted_bboxs_, std::vector<std::vector<std::vector<Dtype> > >
    *min_max_gtruth_, std::vector<std::vector<std::vector<float> > > *iou_);

   /**
    * @brief Do the inverse transformation on the ground truth data
    *       Mathematically, it is the following transformation
    *
    * @param gtruth_ : [N, objects_, 4]
    *   - N        : batch_size
    *   - objects_ : represents the number of objects in an image_width
    *   - 4        : @f$ [center_x, center_y, anchor_height, anchor_width] @f$
    * @param batch_num_objs_ : [N]
    *   - N        : The total number of objects in each image of a batch
    * @param gtruth_inv_ : [N, objs, anchors_, 4]
    *   - N        : batch_size
    *   - objs     : Number of objects in each image
    *   - anchors_ : the total number of anchors `H * W * anchors_per_grid`
    *   - 4        : @f$ [\delta x_{ijk}^{G}, \delta y_{ijk}^{G},
    *                     \delta h_{ijk}^{G}, \delta w_{ijk}^{G}]
    */
    void gtruth_inv_transform(std::vector<std::vector<std::vector<Dtype> > >
    *gtruth_, std::vector<int> *batch_num_objs_, std::vector<std::vector<
    std::vector<std::vector<Dtype> > > > *gtruth_inv_);

};

}  // namespace caffe

#endif  // CAFFE_SQUEEZEDET_LOSS_LAYER_HPP_
