#include <algorithm>
#include <cfloat>
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

    CHECK_EQ(this->layer_param_.squeezedet_param().anchor_shapes_size(),
            2 * anchors_per_grid)
        << "Each anchor must have be specified by two values in the form \
            (width, height)";

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

  // Compute the `center_x` and `center_y` for all the anchors
  // NOTE: Currently supported only for square images
  // Anchor shapes of the form `(center_x, center_y)`
  std::vector<std::pair<float, float> > anchor_center_;
  for (int x = 1; x < W+1; ++x) {
      float c_x = (x * float(image_width)) / (W+1.0);
      float c_y = (x * float(image_height)) / (W+1.0);
      anchor_center_.push_back(std::make_pair(c_x, c_y));
  }

  // Create a 4-d tensor of the form:
  //  @f$ [center_x, center_y, anchor_height, anchor_width] @f$
  anchors_values_.resize(H);
  for (int i = 0; i < H; ++i) {
      anchors_values_[i].resize(W);
      for (int j = 0; j < W; ++j) {
          anchors_values_[i][j].resize(anchors_per_grid);
          for (int k = 0; k < anchors_per_grid; ++k) {
              anchors_values_[i][j][k].resize(4);
          }
      }
  }
  // Do the same for the predictions
  anchors_preds_.resize(H);
  for (int i = 0; i < H; ++i) {
      anchors_preds_[i].resize(W);
      for (int j = 0; j < W; ++j) {
          anchors_preds_[i][j].resize(anchors_per_grid);
          for (int k = 0; k < anchors_per_grid; ++k) {
              anchors_preds_[i][j][k].resize(4);
          }
      }
  }

  // 4d tensor to hold the transformed predictions from ConvDet
  transformed_bbox_preds_.resize(H);
  min_max_pred_bboxs_.resize(H);
  for (int i = 0; i < H; ++i) {
      transformed_bbox_preds_[i].resize(W);
      min_max_pred_bboxs_[i].resize(W);
      for (int j = 0; j < W; ++j) {
          transformed_bbox_preds_[i][j].resize(anchors_per_grid);
          min_max_pred_bboxs_[i][j].resize(anchors_per_grid);
          for (int k = 0; k < anchors_per_grid; ++k) {
              transformed_bbox_preds_[i][j][k].resize(4);
              min_max_pred_bboxs_[i][j][k].resize(4);
          }
      }
  }

  for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
          for (int k = 0; k < anchors_per_grid; ++k) {
              anchors_values_[i][j][k][0] = anchor_center_[i * H + j].first;
              anchors_values_[i][j][k][1] = anchor_center_[i * W + j].second;
              anchors_values_[i][j][k][2] = anchor_shapes_[k].first;
              anchors_values_[i][j][k][3] = anchor_shapes_[k].second;
          }
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

  // Setup the layers
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(softmax_input_vec_);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&probs_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  LayerParameter sigmoid_param(this->layer_param_);
  sigmoid_param.set_type("Sigmoid");
  sigmoid_layer_ = LayerRegistry<Dtype>::CreateLayer(sigmoid_param);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(sigmoid_input_vec_);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(&conf_);
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // separate out the class probability specific values from the input blob
  delete softmax_input_vec_;
  softmax_input_vec_ = new Blob<Dtype>();
  softmax_input_vec_->Reshape(softmax_layer_shape_);
  Dtype* softmax_layer_data_ = softmax_input_vec_->mutable_cpu_data();
  for(size_t batch = 0, idx = 0; batch < N; ++batch) {
    for(size_t height = 0; height < H; ++height) {
      for(size_t width = 0; width < W; ++width) {
        for(size_t ch = 0; ch < anchors_per_grid * classes_; ++ch) {
          softmax_layer_data_[idx] = bottom[0]->data_at(batch, ch, height, width);
          ++idx;
        }
      }
    }
  }
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  // TODO : Reshape the output of softmax (aka `prob_`) to [N, anchors_, classes]
  // TODO : Be sure along which axis softmax is applied

  // separate out the confidence score values from the input blob
  delete sigmoid_input_vec_;
  sigmoid_input_vec_ = new Blob<Dtype>();
  sigmoid_input_vec_->Reshape(sigmoid_layer_shape_);
  Dtype* sigmoid_layer_data_ = sigmoid_input_vec_->mutable_cpu_data();
  for(size_t batch = 0, idx = 0; batch < N; ++batch) {
    for(size_t height = 0; height < H; ++height) {
      for(size_t width = 0; width < W; ++width) {
        for(size_t ch = anchors_per_grid * classes_;
            ch < anchors_per_grid * classes_ + anchors_per_grid * 1; ++ch) {
          sigmoid_layer_data_[idx] =
            bottom[0]->data_at(batch, ch, height, width);
          ++idx;
        }
      }
    }
  }
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // TODO : Reshape the output of sigmoid (aka conf_) to [N, anchors_, 1]

  // separate out the relative bounding box values from the input blob
  delete relative_coord_vec_;
  relative_coord_vec_ = new Blob<Dtype>();
  relative_coord_vec_->Reshape(rel_coord_layer_shape_);
  Dtype* rel_coord_data_ = relative_coord_vec_->mutable_cpu_data();
  for(size_t batch = 0, idx = 0; batch < N; ++batch) {
    for(size_t height = 0; height < H; ++height) {
      for(size_t width = 0; width < W; ++width) {
        for(size_t ch = anchors_per_grid * (classes_ + 1);
            ch < anchors_per_grid * (classes_ + 1 + 4); ++ch) {
          rel_coord_data_[idx] = bottom[0]->data_at(batch, ch, height, width);
          ++idx;
        }
      }
    }
  }
  // TODO : Reshape `rel_coord_vec_` to [N, anchors_, 4]

  // Calculate the number of objects to normalize the regression loss
  int num_objects_ = 0;
  int num_preds = bottom[1]->count();
  const Dtype* preds = bottom[1]->cpu_data();
  for (size_t i = 0; i < num_preds;) {
      num_objects_ += preds[i];
      i += (preds[i] * 5 + 1);
  }
  vector<single_object> bbox_;
  for (size_t i = 0; i < num_preds;) {
    int single_img_objs_count = preds[i];
    for (size_t box = 0; box < single_img_objs_count; ++box) {
      single_object single_bbox;
      single_bbox.xmin = bottom[1]->data_at(i + box * 5 + 1, 1, 1, 1);
      single_bbox.xmax = bottom[1]->data_at(i + box * 5 + 2, 1, 1, 1);
      single_bbox.ymin = bottom[1]->data_at(i + box * 5 + 3, 1, 1, 1);
      single_bbox.ymax = bottom[1]->data_at(i + box * 5 + 4, 1, 1, 1);
      single_bbox.indx = bottom[1]->data_at(i + box * 5 + 5, 1, 1, 1);
      bbox_.push_back(single_bbox);
    }
    i += (preds[i] * 5 + 1);
  }

  for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
          for (int k = 0; k < anchors_per_grid; ++k) {
              anchors_preds_[i][j][k][0] =
                        rel_coord_data_[((i * W + j) * anchors_per_grid + k) * 4 + 0];
              anchors_preds_[i][j][k][1] =
                        rel_coord_data_[((i * W + j) * anchors_per_grid + k) * 4 + 1];
              anchors_preds_[i][j][k][2] =
                        rel_coord_data_[((i * W + j) * anchors_per_grid + k) * 4 + 2];
              anchors_preds_[i][j][k][3] =
                        rel_coord_data_[((i * W + j) * anchors_per_grid + k) * 4 + 3];
          }
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
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      for (int k = 0; k < anchors_per_grid; ++k) {
        transformed_bbox_preds_[i][j][k][0] = anchors_values_[i][j][k][0]
          + anchors_values_[i][j][k][2] * anchors_preds_[i][j][k][0];
        transformed_bbox_preds_[i][j][k][1] = anchors_values_[i][j][k][1]
          + anchors_values_[i][j][k][3] * anchors_preds_[i][j][k][1];

        float delta_w, delta_h;
        caffe::caffe_exp(1, &anchors_preds_[i][j][k][2], &delta_h);
        transformed_bbox_preds_[i][j][k][2] =
          anchors_values_[i][j][k][2] * delta_h;
        caffe::caffe_exp(1, &anchors_preds_[i][j][k][3], &delta_w);
        transformed_bbox_preds_[i][j][k][3] =
          anchors_values_[i][j][k][3] * delta_w;
      }
    }
  }

// Transform the predicted bounding boxes from the form:
// @f$ [center_x, center_y, anchor_height, anchor_width] @f$ to
// @f$ [xmin, ymin, xmax, ymax] @f$
  transform_bbox(transformed_bbox_preds_, min_max_pred_bboxs_);

// Ensure that `(xmin, ymin, xmax, ymax)` are within range of image dimensions
  for (size_t i = 0; i < H; ++i) {
    for (size_t j = 0; j < W; ++j) {
      for (size_t k = 0; k < anchors_per_grid; ++k) {
        Dtype p_xmin = min_max_pred_bboxs_[i][j][k][0];
        Dtype p_ymin = min_max_pred_bboxs_[i][j][k][1];
        Dtype p_xmax = min_max_pred_bboxs_[i][j][k][2];
        Dtype p_ymax = min_max_pred_bboxs_[i][j][k][3];
        min_max_pred_bboxs_[i][j][k][0] = std::min(std::max(Dtype(0.0), p_xmin), image_width-Dtype(1.0));
        min_max_pred_bboxs_[i][j][k][1] = std::min(std::max(Dtype(0.0), p_ymin), image_height-Dtype(1.0));
        min_max_pred_bboxs_[i][j][k][2] = std::max(std::min(image_width-Dtype(1.0), p_xmax), Dtype(0.0));
        min_max_pred_bboxs_[i][j][k][3] = std::max(std::min(image_height-Dtype(1.0), p_ymax), Dtype(0.0));
      }
    }
  }

  // Transform back the valid `(xmin, ymin, xmax, ymax)` to
  // @f$ [center_x, center_y, anchor_height, anchor_width] @f$
  transform_bbox_inv(min_max_pred_bboxs_, transformed_bbox_preds_);

}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::transform_bbox(std::vector<std::vector<
  std::vector<std::vector<Dtype> > > > &pre, std::vector<std::vector<
  std::vector<std::vector<Dtype> > > > &post) {

  for (size_t i = 0; i < H; ++i) {
    for (size_t j = 0; j < W; ++j) {
      for (size_t k = 0; k < anchors_per_grid; ++k) {
        Dtype c_x = pre[i][j][k][0];
        Dtype c_y = pre[i][j][k][1];
        Dtype b_h = pre[i][j][k][2];
        Dtype b_w = pre[i][j][k][3];
        post[i][j][k][0] = c_x - b_w / 2.0;
        post[i][j][k][1] = c_y - b_h / 2.0;
        post[i][j][k][2] = c_x + b_w / 2.0;
        post[i][j][k][3] = c_y + b_h / 2.0;
      }
    }
  }
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::transform_bbox_inv(std::vector<std::vector<
  std::vector<std::vector<Dtype> > > > &pre, std::vector<std::vector<
  std::vector<std::vector<Dtype> > > > &post) {

  for (size_t i = 0; i < H; ++i) {
    for (size_t j = 0; j < W; ++j) {
      for (size_t k = 0; k < anchors_per_grid; ++k) {
        Dtype width = pre[i][j][k][2] - pre[i][j][k][0] + 1.0;
        Dtype height = pre[i][j][k][3] - pre[i][j][k][1] + 1.0;
        post[i][j][k][0] = pre[i][j][k][0] + 0.5 * width;
        post[i][j][k][1] = pre[i][j][k][1] + 0.5 * height;
        post[i][j][k][2] = height;
        post[i][j][k][3] = width;
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

} // namespace caffe
