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
    anchors_     = this->layer_param_.squeezedet_param().anchors();
    classes_     = this->layer_param_.squeezedet_param().classes();
    pos_conf_    = this->layer_param_.squeezedet_param().pos_conf();
    neg_conf_    = this->layer_param_.squeezedet_param().neg_conf();
    lambda_bbox_ = this->layer_param_.squeezedet_param().lambda_bbox();

    CHECK_EQ(this->layer_param_.squeezedet_param().anchor_shapes_size(), 2 * anchors_)
        << "Each anchor must have be specified by two values in the form (width, height)";

    // Anchor shapes of the form `(width, height)`
    for(size_t i = 0; i < this->layer_param_.squeezedet_param().anchor_shapes_size();) {
      const float width  = this->layer_param_.squeezedet_param().anchor_shapes(i);
      const float height = this->layer_param_.squeezedet_param().anchor_shapes(i+1);
      anchor_shapes_.push_back(std::make_pair(width, height));
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
          anchors_values_[i][j].resize(anchors_);
          for (int k = 0; k < anchors_; ++k) {
              anchors_values_[i][j][k].resize(4);
          }
      }
  }

  for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
          for (int k = 0; k < anchors_; ++k) {
              anchors_values_[i][j][k][0] = anchor_center_[i * H + j].first;
              anchors_values_[i][j][k][1] = anchor_center_[i * W + j].second;
              anchors_values_[i][j][k][2] = anchor_shapes_[k].first;
              anchors_values_[i][j][k][3] = anchor_shapes_[k].second;
          }
      }
  }

  // Class specific probability distribution values for each of the anchor
  batch_tot_class_probs = N * H * W * anchors_ * classes_;
  softmax_layer_shape_.push_back(N);
  softmax_layer_shape_.push_back(anchors_ * classes_);
  softmax_layer_shape_.push_back(H);
  softmax_layer_shape_.push_back(W);

  // Confidence Score values for each of the anchor
  batch_tot_conf_scores = N * H * W * anchors_ * 1;
  sigmoid_layer_shape_.push_back(N);
  sigmoid_layer_shape_.push_back(anchors_ * 1);
  sigmoid_layer_shape_.push_back(H);
  sigmoid_layer_shape_.push_back(W);

  // Relative coordinate values for each of the anchor
  batch_tot_rel_coord = N * H * W * anchors_ * 4;
  rel_coord_layer_shape_.push_back(N);
  rel_coord_layer_shape_.push_back(anchors_ * 4);
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
        for(size_t ch = 0; ch < anchors_ * classes_; ++ch) {
          softmax_layer_data_[idx] = bottom[0]->data_at(batch, ch, height, width);
          ++idx;
        }
      }
    }
  }
  // TODO : Reshape the output of softmax to [N, anchors_, classes]
  // TODO : Be sure along which axies softmax is applied

  // separate out the confidence score values from the input blob
  delete sigmoid_input_vec_;
  sigmoid_input_vec_ = new Blob<Dtype>();
  sigmoid_input_vec_->Reshape(sigmoid_layer_shape_);
  Dtype* sigmoid_layer_data_ = sigmoid_input_vec_->mutable_cpu_data();
  for(size_t batch = 0, idx = 0; batch < N; ++batch) {
    for(size_t height = 0; height < H; ++height) {
      for(size_t width = 0; width < W; ++width) {
        for(size_t ch = anchors_ * classes_; ch < anchors_ * classes_ + anchors_ * 1; ++ch) {
          sigmoid_layer_data_[idx] = bottom[0]->data_at(batch, ch, height, width);
          ++idx;
        }
      }
    }
  }
  // TODO : Reshape the output of sigmoid to [N, anchors_, 1]

  // separate out the relative bounding box values from the input blob
  delete relative_coord_vec_;
  relative_coord_vec_ = new Blob<Dtype>();
  relative_coord_vec_->Reshape(rel_coord_layer_shape_);
  Dtype* rel_coord_data_ = relative_coord_vec_->mutable_cpu_data();
  for(size_t batch = 0, idx = 0; batch < N; ++batch) {
    for(size_t height = 0; height < H; ++height) {
      for(size_t width = 0; width < W; ++width) {
        for(size_t ch = anchors_ * (classes_ + 1); ch < anchors_ * (classes_ + 1 + 4); ++ch) {
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

  // Compute the transformation from ConvDet predictions to bounding box predictions
  // @f$ x_{i}^{p} = \hat{x_{i}} + \hat{w_{k}} \delta x_{ijk} @f$
  // @f$ y_{j}^{p} = \hat{y_{j}} + \hat{h_{k}} \delta y_{ijk} @f$
  // @f$ w_{k}^{p} = \hat{w_{k}} \exp{(\delta w_{ijk})} @f$
  // @f$ h_{k}^{p} = \hat{h_{k}} \exp{(\delta h_{ijk})} @f$
  for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {

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
