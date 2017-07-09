#include <algorithm>
#include <cfloat>
#include <utility>
#include <vector>
#include <fstream>

#include "caffe/layers/squeezedet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void  SqueezeDetLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }


  std::vector<std::pair<float, float> > anchor_shapes_;
  if (this->layer_param_.has_squeezedet_param()) {
    anchors_per_grid     = this->layer_param_.squeezedet_param().anchors_per_grid();
    classes_     = this->layer_param_.squeezedet_param().classes();
    pos_conf_    = this->layer_param_.squeezedet_param().pos_conf();
    neg_conf_    = this->layer_param_.squeezedet_param().neg_conf();
    lambda_bbox_ = this->layer_param_.squeezedet_param().lambda_bbox();
    lambda_conf  = this->layer_param_.squeezedet_param().lambda_conf();
    epsilon      = this->layer_param_.squeezedet_param().epsilon();

    CHECK_EQ(this->layer_param_.squeezedet_param().anchor_shapes_size(),
            2 * anchors_per_grid)
        << "Each anchor must be attributed in the form (width, height)";

    // Anchor shapes of the form `(height, width)`
    for (size_t i = 0;
        i < this->layer_param_.squeezedet_param().anchor_shapes_size();) {
      const float width  =
        this->layer_param_.squeezedet_param().anchor_shapes(i);
      const float height =
        this->layer_param_.squeezedet_param().anchor_shapes(i+1);
      anchor_shapes_.push_back(std::make_pair(height, width));
      i += 2;
    }
  } else {
    LOG(FATAL) << "Must specify loss parameters.";
  }

  // Read the shape of the images which are fed to the network
  // TODO: Change this to get the resize_param and not random crop
  if (this->layer_param_.transform_param().crop_size()) {
      image_height = this->layer_param_.transform_param().crop_size();
      image_width = this->layer_param_.transform_param().crop_size();
  } else {
      image_width = 416;
      image_height = 416;
  }

  N = bottom[0]->shape(0);
  H = bottom[0]->shape(1);
  W = bottom[0]->shape(2);
  C = bottom[0]->shape(3);
  // Total Number of anchors in a batch size of `N`:
  anchors_ = H * W * anchors_per_grid;

  // Compute the `center_x` and `center_y` for all the anchors
  // Anchor shapes of the form `(center_x, center_y)`
  std::vector<std::pair<Dtype, Dtype> > anchor_center_;
  for (int x = 1; x < W+1; ++x) {
    Dtype c_x = (x * static_cast<Dtype>(image_width)) / (W+1.0);
    for (int y = 1; y < H+1; ++y) {
      Dtype c_y = (y * static_cast<Dtype>(image_height)) / (H+1.0);
      anchor_center_.push_back(std::make_pair(c_x, c_y));
    }
  }

  // Create a 2-d tensor of the form:
  // @f$ [anchors_, 4] @f$ where the `4` values are as follows:
  // @f$ [center_x, center_y, anchor_height, anchor_width] @f$
  anchors_values_.resize(anchors_);
  for (int i = 0; i < anchors_; ++i) {
      anchors_values_[i].resize(4);
  }
  // Do the same for the predictions
  anchors_preds_.resize(N);
  transformed_bbox_preds_.resize(N);
  min_max_pred_bboxs_.resize(N);
  for (int i = 0; i < N; ++i) {
      anchors_preds_[i].resize(anchors_);
      transformed_bbox_preds_[i].resize(anchors_);
      min_max_pred_bboxs_[i].resize(anchors_);
      for (int j = 0; j < anchors_; ++j) {
        anchors_preds_[i][j].resize(4);
        transformed_bbox_preds_[i][j].resize(4);
        min_max_pred_bboxs_[i][j].resize(4);
      }
  }

  for (int anchor = 0, i = 0, j = 0; anchor < anchors_; ++anchor) {
    anchors_values_[anchor][0] = anchor_center_.at(i).first;
    anchors_values_[anchor][1] = anchor_center_.at(i).second;
    anchors_values_[anchor][2] = anchor_shapes_.at(j).first;
    anchors_values_[anchor][3] = anchor_shapes_.at(j).second;
    if ((anchor+1) % anchors_per_grid == 0) {
      i += 1;
      j = 0;
    } else {
      ++j;
    }
  }

  // Class specific probability distribution values for each of the anchor
  softmax_layer_shape_.push_back(N);
  softmax_layer_shape_.push_back(H * W * anchors_per_grid);
  softmax_layer_shape_.push_back(classes_);

  // Confidence Score values for each of the anchor
  sigmoid_layer_shape_.push_back(N);
  sigmoid_layer_shape_.push_back(H);
  sigmoid_layer_shape_.push_back(W);
  sigmoid_layer_shape_.push_back(anchors_per_grid);

  // Relative coordinate values for each of the anchor
  rel_coord_layer_shape_.push_back(N);
  rel_coord_layer_shape_.push_back(H);
  rel_coord_layer_shape_.push_back(W);
  rel_coord_layer_shape_.push_back(anchors_per_grid * 4);

  // Softmax Layer and it's reshape layer
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  // NHWC is used and not NCHW (default of caffe)
  softmax_param.mutable_softmax_param()->set_axis(2);
  softmax_input_vec_ = new Blob<Dtype>(softmax_layer_shape_);
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(softmax_input_vec_);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&probs_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  LayerParameter reshape_soft_param(this->layer_param_);
  reshape_soft_param.set_type("Reshape");
  reshape_soft_param.clear_reshape_param();
  reshape_soft_param.mutable_reshape_param()->mutable_shape()->add_dim(N);
  reshape_soft_param.mutable_reshape_param()->mutable_shape()->add_dim(anchors_);
  reshape_soft_param.mutable_reshape_param()->mutable_shape()->add_dim(classes_);
  reshape_softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(
      reshape_soft_param);
  reshape_softmax_bottom_vec_.clear();
  reshape_softmax_bottom_vec_.push_back(&probs_);
  reshape_softmax_top_vec_.clear();
  reshape_softmax_top_vec_.push_back(&reshape_probs_);
  reshape_softmax_layer_->SetUp(reshape_softmax_bottom_vec_,
      reshape_softmax_top_vec_);

  // Sigmoid layer and it's reshape layer
  LayerParameter sigmoid_param(this->layer_param_);
  sigmoid_param.set_type("Sigmoid");
  sigmoid_input_vec_ = new Blob<Dtype>(sigmoid_layer_shape_);
  sigmoid_layer_ = LayerRegistry<Dtype>::CreateLayer(sigmoid_param);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(sigmoid_input_vec_);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(&conf_);
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  LayerParameter reshape_sigmoid_param(this->layer_param_);
  reshape_sigmoid_param.set_type("Reshape");
  reshape_sigmoid_param.clear_reshape_param();
  reshape_sigmoid_param.mutable_reshape_param()->mutable_shape()->add_dim(N);
  reshape_sigmoid_param.mutable_reshape_param()->mutable_shape()->add_dim(anchors_);
  reshape_sigmoid_layer_ = LayerRegistry<Dtype>::CreateLayer(
      reshape_sigmoid_param);
  reshape_sigmoid_bottom_vec_.clear();
  reshape_sigmoid_bottom_vec_.push_back(&conf_);
  reshape_sigmoid_top_vec_.clear();
  reshape_sigmoid_top_vec_.push_back(&reshape_conf_);
  reshape_sigmoid_layer_->SetUp(reshape_sigmoid_bottom_vec_,
      reshape_sigmoid_top_vec_);

  // Reshape the relative bbox coordinates
  LayerParameter reshape_bbox_param(this->layer_param_);
  reshape_bbox_param.set_type("Reshape");
  reshape_bbox_param.clear_reshape_param();
  reshape_bbox_param.mutable_reshape_param()->mutable_shape()->add_dim(N);
  reshape_bbox_param.mutable_reshape_param()->mutable_shape()->add_dim(anchors_);
  reshape_bbox_param.mutable_reshape_param()->mutable_shape()->add_dim(4);
  reshape_bbox_layer_ = LayerRegistry<Dtype>::CreateLayer(reshape_bbox_param);
  relative_coord_vec_ = new Blob<Dtype>(rel_coord_layer_shape_);
  reshape_bbox_bottom_vec_.clear();
  reshape_bbox_bottom_vec_.push_back(relative_coord_vec_);
  reshape_bbox_top_vec_.clear();
  reshape_bbox_top_vec_.push_back(&reshape_coord_vec_);
  reshape_bbox_layer_->SetUp(reshape_bbox_bottom_vec_, reshape_bbox_top_vec_);
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  N = bottom[0]->shape(0);
  // Reshape vectors as per the batch size
  gtruth_inv_.resize(N);
  for (size_t batch = 0; batch < N; ++batch) {
    gtruth_inv_[batch].resize(batch_num_objs_[batch]);
    for (size_t obj = 0; obj < batch_num_objs_[batch]; ++obj) {
      gtruth_inv_[batch][obj].resize(anchors_);
      for (size_t anchor = 0; anchor < anchors_; ++anchor) {
        gtruth_inv_[batch][obj][anchor].resize(4);
      }
    }
  }
  final_prob_.resize(N);
  final_conf_.resize(N);
  for (size_t batch = 0; batch < N; ++batch) {
    final_prob_[batch].resize(anchors_);
    final_conf_[batch].resize(anchors_);
    for (size_t anchor = 0; anchor < anchors_; ++anchor) {
      final_prob_[batch][anchor].resize(classes_);
    }
  }
  iou_.resize(N);
  gtruth_.resize(N);
  min_max_gtruth_.resize(N);
  for (size_t i = 0; i < N; ++i) {
    iou_[i].resize(batch_num_objs_[i]);
    gtruth_[i].resize(batch_num_objs_[i]);
    min_max_gtruth_[i].resize(batch_num_objs_[i]);
    for (size_t j = 0; j < batch_num_objs_[i]; ++j) {
      iou_[i][j].resize(anchors_);
      gtruth_[i][j].resize(4);
      min_max_gtruth_[i][j].resize(5);
    }
  }
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  reshape_softmax_layer_->Reshape(reshape_softmax_bottom_vec_, reshape_softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  reshape_sigmoid_layer_->Reshape(reshape_sigmoid_bottom_vec_, reshape_sigmoid_top_vec_);
  reshape_bbox_layer_->Reshape(reshape_bbox_bottom_vec_, reshape_bbox_top_vec_);
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // Calculate the number of objects to normalize the regression loss
  batch_num_objs_.clear();
  const Dtype* label_data = bottom[1]->cpu_data();
  for (size_t batch = 0; batch < N; ++batch) {
    bool flag = false;
    for (size_t j = 0; j < bottom[1]->count() / bottom[1]->shape(0);) {
      if (label_data[batch * bottom[1]->shape(1) + j] == -1) {
        batch_num_objs_.push_back(j / 5);
        flag = true;
        break;
      }
      j += 5;
    }
    if (!flag) {
      batch_num_objs_.push_back(bottom[1]->shape(1) / 5);
    }
    DCHECK_GE(batch_num_objs_[batch], 0);
  }
  CHECK_EQ(batch_num_objs_.size(), N);

  // separate out the class probability specific values from the input blob
  Dtype* softmax_layer_data_ = softmax_input_vec_->mutable_cpu_data();
  for (size_t batch = 0, idx = 0; batch < N; ++batch) {
    for (size_t height = 0; height < H; ++height) {
      for (size_t width = 0; width < W; ++width) {
        for (size_t ch = 0; ch < anchors_per_grid * classes_;
            ++ch) {
          softmax_layer_data_[idx] =
            bottom[0]->data_at(batch, height, width, ch);
          ++idx;
        }
      }
    }
  }
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  reshape_softmax_layer_->Forward(reshape_softmax_bottom_vec_,
      reshape_softmax_top_vec_);
  const Dtype* final_softmax_data = reshape_probs_.cpu_data();

  // separate out the confidence score values from the input blob
  Dtype* sigmoid_layer_data_ = sigmoid_input_vec_->mutable_cpu_data();
  for (size_t batch = 0, idx = 0; batch < N; ++batch) {
    for (size_t height = 0; height < H; ++height) {
      for (size_t width = 0; width < W; ++width) {
        for (size_t ch = anchors_per_grid * classes_;
            ch < anchors_per_grid * classes_ +
            anchors_per_grid * 1; ++ch) {
          sigmoid_layer_data_[idx] =
            bottom[0]->data_at(batch, height, width, ch);
          ++idx;
        }
      }
    }
  }
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  reshape_sigmoid_layer_->Forward(reshape_sigmoid_bottom_vec_,
      reshape_sigmoid_top_vec_);
  const Dtype* final_sigmoid_data = reshape_conf_.cpu_data();

  // separate out the relative bounding box values from the input blob
  Dtype* rel_coord_data_ = relative_coord_vec_->mutable_cpu_data();
  for (size_t batch = 0, idx = 0; batch < N; ++batch) {
    for (size_t height = 0; height < H; ++height) {
      for (size_t width = 0; width < W; ++width) {
        for (size_t ch = anchors_per_grid * (classes_ + 1);
            ch < anchors_per_grid * (classes_ + 1 + 4); ++ch) {
          rel_coord_data_[idx] = bottom[0]->data_at(batch, height, width, ch);
          ++idx;
        }
      }
    }
  }
  reshape_bbox_layer_->Forward(reshape_bbox_bottom_vec_,
      reshape_bbox_top_vec_);
  const Dtype* final_bbox_data_ = reshape_coord_vec_.cpu_data();

  // Extract bounding box data from batch of images
  for (size_t batch = 0; batch < N; ++batch) {
    for (size_t obj = 0; obj < batch_num_objs_[batch]; ++obj) {
      min_max_gtruth_[batch][obj][0] = bottom[1]->data_at(batch, obj * 5
          + 0, 0, 0);
      min_max_gtruth_[batch][obj][1] = bottom[1]->data_at(batch, obj * 5
          + 1, 0, 0);
      min_max_gtruth_[batch][obj][2] = bottom[1]->data_at(batch, obj * 5
          + 2, 0, 0);
      min_max_gtruth_[batch][obj][3] = bottom[1]->data_at(batch, obj * 5
          + 3, 0, 0);
      min_max_gtruth_[batch][obj][4] = bottom[1]->data_at(batch, obj * 5
          + 4, 0, 0);
    }
  }

  // Extract relative coordinates predicted from the `ConvDet` layer
  for (int batch = 0; batch < N; ++batch) {
    for (int anchor = 0; anchor < anchors_; ++anchor) {
      anchors_preds_[batch][anchor][0] = final_bbox_data_[(batch * anchors_
          + anchor) * 4 + 0];
      anchors_preds_[batch][anchor][1] = final_bbox_data_[(batch * anchors_
          + anchor) * 4 + 1];
      anchors_preds_[batch][anchor][2] = final_bbox_data_[(batch * anchors_
          + anchor) * 4 + 2];
      anchors_preds_[batch][anchor][3] = final_bbox_data_[(batch * anchors_
          + anchor) * 4 + 3];
    }
  }

  // Transform relative coordinates from ConvDet to bounding box coordinates
  // @f$ [x_{i}^{p}, y_{j}^{p}, h_{k}^{p}, w_{k}^{p}] @f$
  transform_bbox_predictions(&anchors_preds_, &anchors_values_,
      &transformed_bbox_preds_);

  // Transform each predicted bounding boxes from the form:
  // @f$ [x_{i}^{p}, y_{j}^{p}, h_{k}^{p}, w_{k}^{p}] @f$ to
  // @f$ [xmin, ymin, xmax, ymax] @f$
  transform_bbox(&transformed_bbox_preds_, &min_max_pred_bboxs_);

  // Ensure that `(xmin, ymin, xmax, ymax)` for each of the predicted box
  // are within range of image dimensions
  for (size_t batch = 0; batch < N; ++batch) {
    for (size_t anchor = 0; anchor < anchors_; ++anchor) {
      Dtype p_xmin = min_max_pred_bboxs_[batch][anchor][0];
      Dtype p_ymin = min_max_pred_bboxs_[batch][anchor][1];
      Dtype p_xmax = min_max_pred_bboxs_[batch][anchor][2];
      Dtype p_ymax = min_max_pred_bboxs_[batch][anchor][3];
      min_max_pred_bboxs_[batch][anchor][0] = std::min(std::max(
        static_cast<Dtype>(0.0), p_xmin), image_width -
        static_cast<Dtype>(1.0));
      min_max_pred_bboxs_[batch][anchor][1] = std::min(std::max(
        static_cast<Dtype>(0.0), p_ymin), image_height -
        static_cast<Dtype>(1.0));
      min_max_pred_bboxs_[batch][anchor][2] = std::max(std::min(
        image_width - static_cast<Dtype>(1.0), p_xmax),
        static_cast<Dtype>(0.0));
      min_max_pred_bboxs_[batch][anchor][3] = std::max(std::min(
        image_height - static_cast<Dtype>(1.0), p_ymax),
        static_cast<Dtype>(0.0));
    }
  }

  // Transform back the valid `(xmin, ymin, xmax, ymax)` to
  // @f$ [center_x, center_y, anchor_height, anchor_width] @f$
  transform_bbox_inv(&min_max_pred_bboxs_, &transformed_bbox_preds_);

  // Compute IOU for the predicted boxes and the ground truth
  intersection_over_union(&min_max_pred_bboxs_, &min_max_gtruth_, &iou_);
  // Compute final class specific probabilities

  for (size_t batch = 0; batch < N; ++batch) {
    for (size_t anchor = 0; anchor < anchors_; ++anchor) {
      for (size_t _class = 0; _class < classes_; ++_class) {
        final_prob_[batch][anchor][_class] = final_softmax_data[(batch *
            anchors_ + anchor) * classes_ + _class] * final_sigmoid_data[batch
            * anchors_ + anchor];
      }
    }
  }
  // Store the confidence scores for use during computing gradients
  for (size_t batch = 0; batch < N; ++batch) {
    for (size_t anchor = 0; anchor < anchors_; ++anchor) {
      final_conf_[batch][anchor] = final_sigmoid_data[batch * anchors_ + anchor];
    }
  }

  std::vector<Dtype> class_reg_loss;
  std::vector<Dtype> conf_reg_loss;
  std::vector<Dtype> bbox_reg_loss;
  Dtype loss = 0;

  // Compute the `class regression loss`
  for (size_t batch = 0; batch < N; ++batch) {
    Dtype l = 0;
    for (size_t obj = 0; obj < batch_num_objs_[batch]; ++obj) {
      const int max_iou_indx = std::max_element(iou_[batch][obj].begin(),
          iou_[batch][obj].end()) - iou_[batch][obj].begin();
      const int correct_indx = static_cast<int>(min_max_gtruth_.at(batch).at(obj).at(4));
      DCHECK_GE(correct_indx, 0);
      DCHECK_GE(max_iou_indx, 0);
      DCHECK_LT(correct_indx, classes_);
      DCHECK_LT(max_iou_indx, anchors_);

      // Make sure you don't blow up the loss, add `epsilon` to ensure this
      l -= (log(final_softmax_data[(batch * anchors_ + max_iou_indx) * classes_
          + correct_indx] + epsilon) * lambda_conf);
      }
    class_reg_loss.push_back(l / batch_num_objs_[batch]);
  }

  // Compute the `confidence score regression loss`
  for (size_t batch = 0; batch < N; ++batch) {
    Dtype l1 = 0;
    Dtype l2 = 0;
    for (size_t obj = 0; obj < batch_num_objs_[batch]; ++obj) {
      const int max_iou_indx = std::max_element(iou_[batch][obj].begin(),
          iou_[batch][obj].end()) - iou_[batch][obj].begin();
      DCHECK_GE(max_iou_indx, 0);
      DCHECK_LT(max_iou_indx, anchors_);
      for (size_t anchor = 0; anchor < anchors_; ++anchor) {
        Dtype exp_val, diff_val;
        if (anchor == max_iou_indx) {
          diff_val = final_sigmoid_data[batch * anchors_ + max_iou_indx]
              - iou_[batch][obj][max_iou_indx];
          caffe::caffe_powx(1, &diff_val, static_cast<Dtype>(2.0), &exp_val);
          l1 += exp_val;
        } else {
          diff_val = final_sigmoid_data[batch * anchors_ + anchor];
          caffe::caffe_powx(1, &diff_val, static_cast<Dtype>(2.0), &exp_val);
          l2 += exp_val;
        }
      }
    }
    conf_reg_loss.push_back((pos_conf_ / batch_num_objs_[batch]) * l1
                          + (neg_conf_ / (anchors_ - batch_num_objs_[batch])) * l2);
  }

  // Compute `bounding box regression loss`
  transform_bbox_inv(&min_max_gtruth_, &gtruth_);
  gtruth_inv_transform(&gtruth_, &batch_num_objs_, &gtruth_inv_);
  for (size_t batch = 0; batch < N; ++batch) {
    Dtype l = 0;
    for (size_t obj = 0; obj < batch_num_objs_[batch]; ++obj) {
      const int max_iou_indx = std::max_element(iou_[batch][obj].begin(),
          iou_[batch][obj].end()) - iou_[batch][obj].begin();
      DCHECK_GE(max_iou_indx, 0);
      DCHECK_LT(max_iou_indx, anchors_);
      Dtype exp_val_x, exp_val_y, exp_val_h, exp_val_w;
      Dtype diff_val_x, diff_val_y, diff_val_h, diff_val_w;
      diff_val_x = final_bbox_data_[(batch * anchors_ + max_iou_indx) * 4
            + 0] - gtruth_inv_[batch][obj][max_iou_indx][0];
      diff_val_y = final_bbox_data_[(batch * anchors_ + max_iou_indx) * 4
            + 1] - gtruth_inv_[batch][obj][max_iou_indx][1];
      diff_val_h = final_bbox_data_[(batch * anchors_ + max_iou_indx) * 4
            + 2] - gtruth_inv_[batch][obj][max_iou_indx][2];
      diff_val_w = final_bbox_data_[(batch * anchors_ + max_iou_indx) * 4
            + 3] - gtruth_inv_[batch][obj][max_iou_indx][3];
      caffe::caffe_powx(1, &diff_val_x, static_cast<Dtype>(2.0), &exp_val_x);
      caffe::caffe_powx(1, &diff_val_y, static_cast<Dtype>(2.0), &exp_val_y);
      caffe::caffe_powx(1, &diff_val_h, static_cast<Dtype>(2.0), &exp_val_h);
      caffe::caffe_powx(1, &diff_val_w, static_cast<Dtype>(2.0), &exp_val_w);
      l += (exp_val_x + exp_val_y + exp_val_h + exp_val_w);
    }
    bbox_reg_loss.push_back((l * lambda_bbox_) / (batch_num_objs_[batch]));
  }

  for (typename std::vector<Dtype>::iterator itr = class_reg_loss.begin();
      itr != class_reg_loss.end(); ++itr) {
    loss += (*itr);
  }
  for (typename std::vector<Dtype>::iterator itr = conf_reg_loss.begin();
      itr != conf_reg_loss.end(); ++itr) {
    loss += (*itr);
  }
  for (typename std::vector<Dtype>::iterator itr = bbox_reg_loss.begin();
      itr != bbox_reg_loss.end(); ++itr) {
    loss += (*itr);
  }
  // TODO : Use normalization param from protobuf message to normalize loss
  top[0]->mutable_cpu_data()[0] = loss / N;
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::transform_bbox(std::vector<std::vector<
    std::vector<Dtype> > > *pre, std::vector<std::vector<
    std::vector<Dtype> > > *post) {

  for (size_t batch = 0; batch < (*pre).size(); ++batch) {
    for (size_t anchor = 0; anchor < (*pre)[batch].size(); ++anchor) {
      Dtype c_x = (*pre)[batch][anchor][0];
      Dtype c_y = (*pre)[batch][anchor][1];
      Dtype b_h = (*pre)[batch][anchor][2];
      Dtype b_w = (*pre)[batch][anchor][3];
      (*post)[batch][anchor][0] = c_x - b_w / 2.0;
      (*post)[batch][anchor][1] = c_y - b_h / 2.0;
      (*post)[batch][anchor][2] = c_x + b_w / 2.0;
      (*post)[batch][anchor][3] = c_y + b_h / 2.0;
    }
  }
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::transform_bbox_inv(std::vector<std::vector<
    std::vector<Dtype> > > *pre, std::vector<std::vector<
    std::vector<Dtype> > > *post) {

  for (size_t batch = 0; batch < (*pre).size(); ++batch) {
    for (size_t anchor = 0; anchor < (*pre)[batch].size(); ++anchor) {
      Dtype width = (*pre)[batch][anchor][2] - (*pre)[batch][anchor][0] + 1.0;
      Dtype height = (*pre)[batch][anchor][3] - (*pre)[batch][anchor][1] + 1.0;
      (*post)[batch][anchor][0] = (*pre)[batch][anchor][0] + 0.5 * width;
      (*post)[batch][anchor][1] = (*pre)[batch][anchor][1] + 0.5 * height;
      (*post)[batch][anchor][2] = height;
      (*post)[batch][anchor][3] = width;
    }
  }
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::intersection_over_union(std::vector<
    std::vector<std::vector<Dtype> > > *predicted_bboxs_, std::vector<
    std::vector<std::vector<Dtype> > > *min_max_gtruth_, std::vector<
    std::vector<std::vector<float> > > *iou_) {

  for (size_t batch = 0; batch < N; ++batch) {
    for (size_t obj = 0; obj < (*min_max_gtruth_)[batch].size(); ++obj) {
      for (size_t anchor = 0; anchor < anchors_; ++anchor) {
        Dtype xmin = std::max((*predicted_bboxs_)[batch][anchor][0],
            (*min_max_gtruth_)[batch][obj][0]);
        Dtype ymin = std::max((*predicted_bboxs_)[batch][anchor][1],
            (*min_max_gtruth_)[batch][obj][1]);
        Dtype xmax = std::min((*predicted_bboxs_)[batch][anchor][2],
            (*min_max_gtruth_)[batch][obj][2]);
        Dtype ymax = std::min((*predicted_bboxs_)[batch][anchor][3],
            (*min_max_gtruth_)[batch][obj][3]);

        // Compute the intersection
        float w = std::max(static_cast<Dtype>(0.0), xmax - xmin);
        float h = std::max(static_cast<Dtype>(0.0), ymax - ymin);
        float inter_ = w * h;
        // Compute the union
        float w_p = (*predicted_bboxs_)[batch][anchor][2]
            - (*predicted_bboxs_)[batch][anchor][0];
        float h_p = (*predicted_bboxs_)[batch][anchor][3]
            - (*predicted_bboxs_)[batch][anchor][1];
        float w_g = (*min_max_gtruth_)[batch][obj][2]
            - (*min_max_gtruth_)[batch][obj][0];
        float h_g = (*min_max_gtruth_)[batch][obj][3]
            - (*min_max_gtruth_)[batch][obj][1];
        float union_ = w_p * h_p + w_g * h_g - inter_;

        // Don't blow up `IOU`, so adding `epsilon`
        (*iou_)[batch][obj][anchor] = inter_ / (union_ + epsilon);
      }
    }
  }
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::gtruth_inv_transform(std::vector<std::vector<
    std::vector<Dtype> > > *gtruth_, std::vector<int> *batch_num_objs_,
    std::vector<std::vector<std::vector<std::vector<Dtype> > > >
    *gtruth_inv_) {

  for (size_t batch = 0; batch < N; ++batch) {
    for (size_t anchor = 0; anchor < anchors_; ++anchor) {
      Dtype x_hat = anchors_values_[anchor][0];
      Dtype y_hat = anchors_values_[anchor][1];
      Dtype h_hat = anchors_values_[anchor][2];
      Dtype w_hat = anchors_values_[anchor][3];
      for (size_t obj = 0; obj < (*batch_num_objs_)[batch]; ++obj) {
        Dtype x_g = (*gtruth_)[batch][obj][0];
        Dtype y_g = (*gtruth_)[batch][obj][1];
        Dtype h_g = (*gtruth_)[batch][obj][2];
        Dtype w_g = (*gtruth_)[batch][obj][3];

        (*gtruth_inv_)[batch][obj][anchor][0] = (x_g - x_hat) / w_hat;
        (*gtruth_inv_)[batch][obj][anchor][1] = (y_g - y_hat) / h_hat;
        (*gtruth_inv_)[batch][obj][anchor][2] = log((h_g / h_hat) + epsilon);
        (*gtruth_inv_)[batch][obj][anchor][3] = log((w_g / w_hat) + epsilon);
      }
    }
  }
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::transform_bbox_predictions(
    std::vector<std::vector<std::vector<Dtype> > > *anchors_preds_,
    std::vector<std::vector<Dtype> > *anchors_values_,
    std::vector<std::vector<std::vector<Dtype> > > *transformed_bbox_preds_) {
  for (int batch = 0; batch < N; ++batch) {
    for (int anchor = 0; anchor < anchors_; ++anchor) {
      Dtype delta_h, delta_w;
      caffe::caffe_exp(1, &(*anchors_preds_)[batch][anchor][2], &delta_h);
      caffe::caffe_exp(1, &(*anchors_preds_)[batch][anchor][3], &delta_w);
      (*transformed_bbox_preds_)[batch][anchor][0] = (*anchors_values_)[anchor][0]
        + (*anchors_values_)[anchor][3] * (*anchors_preds_)[batch][anchor][0];
      (*transformed_bbox_preds_)[batch][anchor][1] = (*anchors_values_)[anchor][1]
        + (*anchors_values_)[anchor][2] * (*anchors_preds_)[batch][anchor][1];
      (*transformed_bbox_preds_)[batch][anchor][2] = (*anchors_values_)[anchor][2]
        * delta_h;
      (*transformed_bbox_preds_)[batch][anchor][3] = (*anchors_values_)[anchor][3]
        * delta_w;
    }
  }
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    CHECK_EQ(bottom[0]->count(), N * H * W * C);
    for (size_t batch = 0; batch < N; ++batch) {
      for (size_t obj = 0; obj < batch_num_objs_[batch]; ++obj) {
        const int max_iou_indx = std::max_element(iou_[batch][obj].begin(),
             iou_[batch][obj].end()) - iou_[batch][obj].begin();
        Dtype class_idx = min_max_gtruth_[batch][obj][4];
        DCHECK_GE(max_iou_indx, 0);
        DCHECK_LT(max_iou_indx, anchors_);
        for (size_t height = 0; height < H; ++height) {
          for (size_t width = 0; width < W; ++width) {
            for (size_t ch = 0; ch < C; ++ch) {
              Dtype _grad;
              if (ch < anchors_per_grid * classes_) {
                // GRADIENTS FOR CROSS ENTROPY LOSS
                const int anchor_idx = (height * W + width) * anchors_per_grid
                     + (ch % classes_);
                if (anchor_idx == max_iou_indx) {
                  const int class_id = ch % classes_;
                  Dtype class_prob_val_ =
                       final_prob_[batch][anchor_idx][class_id];
                  const int label_value = static_cast<int>(class_idx);
                  if (label_value == class_id) {
                    _grad = (1 - class_prob_val_) / batch_num_objs_[batch];
                  } else {
                    _grad = (-class_prob_val_) / batch_num_objs_[batch];
                  }
                } else {
                  _grad = 0;
                }
              } else if (ch >= anchors_per_grid * classes_ &&
                         ch < anchors_per_grid * (classes_ + 1)) {
                // GRADIENTS FOR CLASS REGRESSION LOSS
                const int anchor_idx = (height * W + width) * anchors_per_grid
                     + (ch % (anchors_per_grid * classes_ + 1));
                if (anchor_idx == max_iou_indx) {
                  // TODO : Compute gradients
                  _grad = 0;
                } else {
                  _grad = 0;
                }
              } else {
                // GRADIENTS FOR BBOX REGRESSION LOSS
                const int anchor_idx = (height * W + width) * anchors_per_grid
                     + (ch % (anchors_per_grid * (classes_ + 1) + 4));
                if (anchor_idx == max_iou_indx) {
                  // TODO : Compute gradients
                  _grad = 0;
                } else {
                  _grad = 0;
                }
              }
              bottom_diff[((batch * H + height) * W + width) * C + ch] += _grad;
            }
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SqueezeDetLossLayer);
#endif

INSTANTIATE_CLASS(SqueezeDetLossLayer);
REGISTER_LAYER_CLASS(SqueezeDetLoss);

}  // namespace caffe
