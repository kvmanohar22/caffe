#include <algorithm>
#include <cfloat>
#include <utility>
#include <vector>

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
    anchors_per_grid     = this->layer_param_.squeezedet_param().anchors();
    classes_     = this->layer_param_.squeezedet_param().classes();
    pos_conf_    = this->layer_param_.squeezedet_param().pos_conf();
    neg_conf_    = this->layer_param_.squeezedet_param().neg_conf();
    lambda_bbox_ = this->layer_param_.squeezedet_param().lambda_bbox();
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
      LOG(FATAL) << "Must specify the crop-size in `transform_param`.";
  }

  N = bottom[0]->shape(0);
  C = bottom[0]->shape(1);
  H = bottom[0]->shape(2);
  W = bottom[0]->shape(3);
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

  for (int anchor = 0, i = 0, j = 0; anchor < anchors_; ++anchor, ++j) {
    anchors_values_[anchor][0] = anchor_center_[i].first;
    anchors_values_[anchor][1] = anchor_center_[i].second;
    anchors_values_[anchor][2] = anchor_shapes_[j].first;
    anchors_values_[anchor][3] = anchor_shapes_[j].second;
    if (i == (anchors_per_grid-1)) {
      i += 1;
      j = 0;
    }
  }

  // Class specific probability distribution values for each of the anchor
  batch_tot_class_probs = N * H * W * anchors_per_grid * classes_;
  softmax_layer_shape_.push_back(N);
  softmax_layer_shape_.push_back(anchors_per_grid * classes_);
  softmax_layer_shape_.push_back(H);
  softmax_layer_shape_.push_back(W);

  // Confidence Score values for each of the anchor
  batch_tot_conf_scores = N * H * W * anchors_per_grid * 1;
  sigmoid_layer_shape_.push_back(N);
  sigmoid_layer_shape_.push_back(anchors_per_grid * 1);
  sigmoid_layer_shape_.push_back(H);
  sigmoid_layer_shape_.push_back(W);

  // Relative coordinate values for each of the anchor
  batch_tot_rel_coord = N * H * W * anchors_per_grid * 4;
  rel_coord_layer_shape_.push_back(N);
  rel_coord_layer_shape_.push_back(anchors_per_grid * 4);
  rel_coord_layer_shape_.push_back(H);
  rel_coord_layer_shape_.push_back(W);

  // Softmax Layer and it's reshape layer
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(softmax_input_vec_);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&probs_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  // Reshape the output of softmax layer
  LayerParameter reshape_soft_param(this->layer_param_);
  reshape_soft_param.set_type("Reshape");
  reshape_soft_param.clear_reshape_param();
  reshape_soft_param.mutable_reshape_param()->mutable_shape()->set_dim(0,
      N);
  reshape_soft_param.mutable_reshape_param()->mutable_shape()->set_dim(1,
      anchors_);
  reshape_soft_param.mutable_reshape_param()->mutable_shape()->set_dim(2,
      classes_);
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
  sigmoid_layer_ = LayerRegistry<Dtype>::CreateLayer(sigmoid_param);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(sigmoid_input_vec_);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(&conf_);
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Reshape the output of sigmoid layer
  LayerParameter reshape_sigmoid_param(this->layer_param_);
  reshape_sigmoid_param.set_type("Reshape");
  reshape_sigmoid_param.clear_reshape_param();
  reshape_sigmoid_param.mutable_reshape_param()->mutable_shape()->set_dim(0,
      N);
  reshape_sigmoid_param.mutable_reshape_param()->mutable_shape()->set_dim(1,
      anchors_);
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
  reshape_bbox_param.mutable_reshape_param()->mutable_shape()->set_dim(0,
      N);
  reshape_bbox_param.mutable_reshape_param()->mutable_shape()->set_dim(1,
      anchors_);
  reshape_bbox_param.mutable_reshape_param()->mutable_shape()->set_dim(2,
      4);
  reshape_bbox_layer_ = LayerRegistry<Dtype>::CreateLayer(reshape_bbox_param);
  reshape_bbox_bottom_vec_.clear();
  reshape_bbox_bottom_vec_.push_back(relative_coord_vec_);
  reshape_bbox_top_vec_.clear();
  reshape_bbox_top_vec_.push_back(&reshape_coord_vec_);
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // separate out the class probability specific values from the input blob
  softmax_input_vec_ = new Blob<Dtype>();
  softmax_input_vec_->Reshape(softmax_layer_shape_);
  Dtype* softmax_layer_data_ = softmax_input_vec_->mutable_cpu_data();
  for (size_t batch = 0, idx = 0; batch < N; ++batch) {
    for (size_t height = 0; height < H; ++height) {
      for (size_t width = 0; width < W; ++width) {
        for (size_t ch = 0; ch < anchors_per_grid * classes_; ++ch) {
          softmax_layer_data_[idx] =
            bottom[0]->data_at(batch, ch, height, width);
          ++idx;
        }
      }
    }
  }
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  reshape_softmax_layer_->Forward(reshape_softmax_bottom_vec_,
      reshape_softmax_top_vec_);
  const Dtype* final_softmax_data = reshape_probs_.cpu_data();
  // TODO : Be sure along which axis softmax is applied

  // separate out the confidence score values from the input blob
  sigmoid_input_vec_ = new Blob<Dtype>();
  sigmoid_input_vec_->Reshape(sigmoid_layer_shape_);
  Dtype* sigmoid_layer_data_ = sigmoid_input_vec_->mutable_cpu_data();
  for (size_t batch = 0, idx = 0; batch < N; ++batch) {
    for (size_t height = 0; height < H; ++height) {
      for (size_t width = 0; width < W; ++width) {
        for (size_t ch = anchors_per_grid * classes_;
            ch < anchors_per_grid * classes_ + anchors_per_grid * 1; ++ch) {
          sigmoid_layer_data_[idx] =
            bottom[0]->data_at(batch, ch, height, width);
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
  relative_coord_vec_ = new Blob<Dtype>();
  relative_coord_vec_->Reshape(rel_coord_layer_shape_);
  Dtype* rel_coord_data_ = relative_coord_vec_->mutable_cpu_data();
  for (size_t batch = 0, idx = 0; batch < N; ++batch) {
    for (size_t height = 0; height < H; ++height) {
      for (size_t width = 0; width < W; ++width) {
        for (size_t ch = anchors_per_grid * (classes_ + 1);
            ch < anchors_per_grid * (classes_ + 1 + 4); ++ch) {
          rel_coord_data_[idx] = bottom[0]->data_at(batch, ch, height, width);
          ++idx;
        }
      }
    }
  }
  reshape_bbox_layer_->Forward(reshape_bbox_bottom_vec_,
      reshape_bbox_top_vec_);
  const Dtype* final_bbox_data_ = reshape_coord_vec_.cpu_data();

  // Calculate the number of objects to normalize the regression loss
  std::vector<int> batch_num_objs_;
  int num_preds = bottom[1]->count();
  const Dtype* preds = bottom[1]->cpu_data();
  for (size_t i = 0; i < num_preds;) {
      batch_num_objs_.push_back(preds[i]);
      i += (preds[i] * 5 + 1);
  }
  CHECK_EQ(N, batch_num_objs_.size());

  std::vector<std::vector<std::vector<Dtype> > > gtruth_, min_max_gtruth_;
  gtruth_.resize(N);
  min_max_gtruth_.resize(N);
  for (size_t i = 0; i < N; ++i) {
    gtruth_[i].resize(batch_num_objs_[i]);
    min_max_gtruth_[i].resize(batch_num_objs_[i]);
    for (size_t j = 0; j < batch_num_objs_[i]; ++j) {
      gtruth_[i][j].resize(5);
      min_max_gtruth_[i][j].resize(4);
    }
  }
  for (size_t batch = 0, i = 0; batch < N; ++batch) {
    for (size_t obj = 0; obj < batch_num_objs_[batch]; ++obj) {
      gtruth_[batch][obj][0] = bottom[1]->data_at(i + obj * 5 + 1, 1, 1, 1);
      gtruth_[batch][obj][1] = bottom[1]->data_at(i + obj * 5 + 2, 1, 1, 1);
      gtruth_[batch][obj][2] = bottom[1]->data_at(i + obj * 5 + 3, 1, 1, 1);
      gtruth_[batch][obj][3] = bottom[1]->data_at(i + obj * 5 + 4, 1, 1, 1);
      gtruth_[batch][obj][4] = bottom[1]->data_at(i + obj * 5 + 5, 1, 1, 1);
    }
    i += (batch_num_objs_[batch] * 5 + 1);
  }

// Compute relative coordinates predicted from the `ConvDet` layer
  for (int batch = 0; batch < N; ++batch) {
    for (int anchor = 0; anchor < anchors_; ++anchor) {
      anchors_preds_[batch][anchor][0] = final_bbox_data_[(batch * anchors_
          + anchor) + 0];
      anchors_preds_[batch][anchor][1] = final_bbox_data_[(batch * anchors_
          + anchor) + 1];
      anchors_preds_[batch][anchor][2] = final_bbox_data_[(batch * anchors_
          + anchor) + 2];
      anchors_preds_[batch][anchor][3] = final_bbox_data_[(batch * anchors_
          + anchor) + 3];
    }
  }


  // Compute bounding box predictions from ConvDet predictions
  // Bounding box predictions are of the following form:
  // @f$ [center_x, center_y, anchor_height, anchor_width] @f$
  // Transformation is as follows:
  // @f$ x_{i}^{p} = \hat{x_{i}} + \hat{w_{k}} \delta x_{ijk} @f$
  // @f$ y_{j}^{p} = \hat{y_{j}} + \hat{h_{k}} \delta y_{ijk} @f$
  // @f$ h_{k}^{p} = \hat{h_{k}} \exp{(\delta h_{ijk})} @f$
  // @f$ w_{k}^{p} = \hat{w_{k}} \exp{(\delta w_{ijk})} @f$
  for (int batch = 0; batch < N; ++batch) {
    for (int anchor = 0; anchor < anchors_; ++anchor) {
      Dtype delta_h, delta_w;
      caffe::caffe_exp(1, &anchors_preds_[batch][anchor][2], &delta_h);
      caffe::caffe_exp(1, &anchors_preds_[batch][anchor][3], &delta_w);
      transformed_bbox_preds_[batch][anchor][0] = anchors_values_[anchor][0]
          + anchors_values_[anchor][3] * anchors_preds_[batch][anchor][0];
      transformed_bbox_preds_[batch][anchor][1] = anchors_values_[anchor][1]
          + anchors_values_[anchor][2] * anchors_preds_[batch][anchor][1];
      transformed_bbox_preds_[batch][anchor][2] = anchors_values_[anchor][2]
          * delta_h;
      transformed_bbox_preds_[batch][anchor][3] = anchors_values_[anchor][3]
          * delta_w;
    }
  }

// Transform each predicted bounding boxes from the form:
// @f$ [center_x, center_y, anchor_height, anchor_width] @f$ to
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

  std::vector<std::vector<std::vector<float> > > iou_;
  iou_.resize(N);
  for (size_t i = 0; i < N; ++i) {
    iou_[i].resize(anchors_);
    for (size_t j = 0; j < anchors_; ++j) {
      iou_[i][j].resize(batch_num_objs_[i]);
    }
  }

  transform_bbox(&gtruth_, &min_max_gtruth_);
  // Compute IOU for the predicted boxes and the ground truth
  intersection_over_union(&min_max_pred_bboxs_, &min_max_gtruth_, &iou_);

  // Compute final class specific probabilities
  std::vector<std::vector<std::vector<Dtype> > > final_prob_;
  final_prob_.resize(N);
  for (size_t batch = 0; batch < N; ++batch) {
    final_prob_[batch].resize(anchors_);
    for (size_t anchor = 0; anchor < anchors_; ++anchor) {
      final_prob_[batch][anchor].resize(classes_);
    }
  }
  for (size_t batch = 0; batch < N; ++batch) {
    for (size_t anchor = 0; anchor < anchors_; ++anchor) {
      for (size_t _class = 0; _class < classes_; ++_class) {
        final_prob_[batch][anchor][_class] = final_softmax_data[(batch *
            anchors_ + anchor) * classes_ + _class] * final_sigmoid_data[batch
            * anchors_ + anchor];
      }
    }
  }
  // Free memory
  delete softmax_input_vec_;
  delete sigmoid_input_vec_;
  delete relative_coord_vec_;
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
    std::vector<std::vector<Dtype> > > *gtruth_, std::vector<std::vector<
    std::vector<float> > > *iou_) {

  for (size_t batch = 0; batch < N; ++batch) {
    for (size_t anchor = 0; anchor < anchors_; ++anchor) {
      for (size_t gtruth = 0; gtruth < (*gtruth_)[batch].size(); ++gtruth) {
        Dtype xmin = std::max((*predicted_bboxs_)[batch][anchor][0],
                              (*gtruth_)[batch][gtruth][0]);
        Dtype ymin = std::max((*predicted_bboxs_)[batch][anchor][1],
                              (*gtruth_)[batch][gtruth][1]);
        Dtype xmax = std::min((*predicted_bboxs_)[batch][anchor][2],
                              (*gtruth_)[batch][gtruth][2]);
        Dtype ymax = std::min((*predicted_bboxs_)[batch][anchor][3],
                              (*gtruth_)[batch][gtruth][3]);

        // Compute the intersection
        float w = std::max(static_cast<Dtype>(0.0), xmax - xmin);
        float h = std::max(static_cast<Dtype>(0.0), ymax - ymin);
        float inter_ = w * h;
        // Compute the union
        float w_p = (*predicted_bboxs_)[batch][anchor][2]
                    - (*predicted_bboxs_)[batch][anchor][0];
        float h_p = (*predicted_bboxs_)[batch][anchor][3]
                    - (*predicted_bboxs_)[batch][anchor][1];
        float w_g = (*gtruth_)[batch][gtruth][2] - (*gtruth_)[batch][gtruth][0];
        float h_g = (*gtruth_)[batch][gtruth][3] - (*gtruth_)[batch][gtruth][1];
        float union_ = w_p * h_p + w_g * h_g - inter_;
        (*iou_)[batch][anchor][gtruth] = inter_ / (union_ + epsilon);
      }
    }
  }
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(SqueezeDetLossLayer);
#endif

INSTANTIATE_CLASS(SqueezeDetLossLayer);
REGISTER_LAYER_CLASS(SqueezeDetLoss);

}  // namespace caffe
