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
  
  if(!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE; 
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  if(this->layer_param_.has_squeezedet_param()) {
    anchors_     = this->layer_param_.squeezedet_param().anchors();
    classes_     = this->layer_param_.squeezedet_param().classes();
    pos_conf_    = this->layer_param_.squeezedet_param().pos_conf();
    neg_conf_    = this->layer_param_.squeezedet_param().neg_conf();
    lambda_bbox_ = this->layer_param_.squeezedet_param().lambda_bbox();

    for(size_t i = 0; i < this->layer_param_.squeezedet_param().anchor_shapes_size();) {
      const unsigned int width  = this->layer_param_.squeezedet_param().anchor_shapes[i];
      const unsigned int height = this->layer_param_.squeezedet_param().anchor_shapes[i+1];
      anchor_shapes_.push_back(std::make_pair(width, height));
      i += 2;
    }
  } else {
    // Raise an exception 
    // Should specify the values
  }
  N = bottom[0]->shape(0);
  C = bottom[0]->shape(1);
  H = bottom[0]->shape(2);
  W = bottom[0]->shape(3);

  // separate out the class probability specific values from the input blob
  batch_tot_class_probs = N * H * W * anchors_ * classes_;
  softmax_layer_shape_.push_back(N);
  softmax_layer_shape_.push_back(anchors_ * classes_);
  softmax_layer_shape_.push_back(H);
  softmax_layer_shape_.push_back(W);
  softmax_layer_data_ = new Dtype[batch_tot_class_probs];
  softmax_input_vec_ = new Blob<Dtype>(softmax_layer_shape_);
  for(size_t batch=0, idx=0; batch < N; ++batch){
    for(size_t height=0; height < H; ++height){
      for(size_t width=0; width < W; ++width){
        for(size_t ch=0; ch < anchors_ * classes_; ++ch){
          softmax_layer_data_[idx] = bottom[0]->data_at(batch, ch, height, width);
          idx++;
        }
      }
    }
  }
  softmax_input_vec_->set_cpu_data(softmax_layer_data_);
  // TODO : Reshape the output of softmax to [N, anchors_, classes]
  // TODO : Be sure along which axies softmax is applied

  // separate out the confidence score values from the input blob
  batch_tot_conf_scores = N * H * W * anchors_ * 1;
  sigmoid_layer_shape_.push_back(N);
  sigmoid_layer_shape_.push_back(anchors_ * 1);
  sigmoid_layer_shape_.push_back(H);
  sigmoid_layer_shape_.push_back(W);
  sigmoid_layer_data_ = new Dtype[batch_tot_conf_scores];
  sigmoid_input_vec_ = new Blob<Dtype>(sigmoid_layer_shape_);
  for(size_t batch=0, idx=0; batch < N; ++batch){
    for(size_t height=0; height < H; ++height){
      for(size_t width=0; width < W; ++width){
        for(size_t ch=anchors_ * classes_; ch < anchors_ * classes_ + anchors_ * 1; ++ch){
          sigmoid_layer_data_[idx] = bottom[0]->data_at(batch, ch, height, width);
          idx++;
        }
      }
    }
  }
  sigmoid_input_vec_->set_cpu_data(sigmoid_layer_data_);
  // TODO : Reshape the output of sigmoid to [N, anchors_, 1]

  // separate out the relative bounding box values from the input blob
  batch_tot_rel_coord = N * H * W * anchors_ * 4;
  rel_coord_layer_shape_.push_back(N);
  rel_coord_layer_shape_.push_back(anchors_ * 4);
  rel_coord_layer_shape_.push_back(H);
  rel_coord_layer_shape_.push_back(W);
  rel_coord_data_ = new Dtype[batch_tot_rel_coord];
  relative_coord_vec_ = new Blob<Dtype>(rel_coord_layer_shape_);
  for(size_t batch=0, idx=0, batch < N; ++batch){
    for(size_t height=0; height < H; ++height){
      for(size_t width=0; width < W; ++width){
        for(size_t ch=anchors_ * (classes_ + 1); ch < anchors_ * (classes_ + 1 + 4); ++ch){
          rel_coord_data_[idx] = bottom[0]->data_at(batch, ch, height, width);
          idx++;
        }
      }
    }
  }
  relative_coord_vec_->set_cpu_data(rel_coord_data_);
  // TODO : Reshape `rel_coord_vec_` to [N, anchors_, 4]

  // Setup the layers
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(softmax_input_vec_);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&probs_);
  softmax_layer_.SetUp(softmax_bottom_vec_, softmax_top_vec_);

  LayerParameter sigmoid_param(this->layer_param_);
  sigmoid_param.set_type("Sigmoid");
  sigmoid_layer_ = LayerRegistry<Dtype>::CreateLayer(sigmoid_param);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(sigmoid_input_vec_);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(&conf_);
  sigmoid_layer_.SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  // Calculate the number of objects to normalize the regression loss
  num_objects_ = bottom[1]-> ; // TODO
  for(size_t box=0; box<num_objects_; ++box){
    object singe_bbox;
    single_bbox.xmin = bottom[1]->
    single_bbox.xmax = bottom[1]->
    single_bbox.ymin = bottom[1]->
    single_bbox.ymax = bottom[1]->
    bbox_.push_back(single_bbox);
  }
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

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