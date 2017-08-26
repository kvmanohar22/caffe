#include <algorithm>
#include <cfloat>
#include <numeric>
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
    anchors_per_grid = this->layer_param_.squeezedet_param().anchors_per_grid();
    classes_         = this->layer_param_.squeezedet_param().classes();
    pos_conf_        = this->layer_param_.squeezedet_param().pos_conf();
    neg_conf_        = this->layer_param_.squeezedet_param().neg_conf();
    lambda_bbox_     = this->layer_param_.squeezedet_param().lambda_bbox();
    lambda_conf      = this->layer_param_.squeezedet_param().lambda_conf();
    epsilon          = this->layer_param_.squeezedet_param().epsilon();
    n_top_detections = this->layer_param_.squeezedet_param().n_top_detections();
    intersection_thresh = \
        this->layer_param_.squeezedet_param().intersection_thresh();

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
  if (this->layer_param_.transform_param().crop_size()) {
      image_height = this->layer_param_.transform_param().crop_size();
      image_width = this->layer_param_.transform_param().crop_size();
  } else {
      image_width = 416;
      image_height = 416;
  }

  N = bottom[1]->shape(0);
  H = bottom[1]->shape(1);
  W = bottom[1]->shape(2);
  C = bottom[1]->shape(3);
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
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  N = bottom[1]->shape(0);
  H = bottom[1]->shape(1);
  W = bottom[1]->shape(2);
  C = bottom[1]->shape(3);
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // LOG(INFO) << "Bbox length: " << bottom[3]->shape(0);
  std::vector<std::vector<std::vector<Dtype> > > final_prob_;

  // Transformed  `ConvDet` predictions to bounding boxes which are of shape:
  // @f$ [N, anchors_, 4] @f$ where the `4` values are as follows:
  // @f$ [center_x, center_y, anchor_height, anchor_width] @f$
  std::vector<std::vector<std::vector<Dtype> > > transformed_bbox_preds_;

  // Transformed predicted bounding boxes from:
  // @f$ [N, anchors_, 4] @f$ where the `4` values are as follows:
  // @f$ [center_x, center_y, anchor_height, anchor_width] @f$ to
  // @f$ [N, anchors_, 4] @f$ where the `4` values are as follows:
  // @f$ [xmin, ymin, xmax, ymax] @f$
  std::vector<std::vector<std::vector<std::vector<Dtype> > > > gtruth_inv_;
  std::vector<std::vector<std::vector<Dtype> > > min_max_pred_bboxs_;
  std::vector<std::vector<std::vector<Dtype> > > gtruth_, min_max_gtruth_;
  std::vector<std::vector<std::vector<float> > > iou_;

  const Dtype* class_scores = bottom[0]->cpu_data();
  const Dtype* conf_scores  = bottom[1]->cpu_data();
  const Dtype* delta_bbox   = bottom[2]->cpu_data();
  const Dtype* label_data   = bottom[3]->cpu_data();

  // Calculate the number of objects in each image of a batch of data
  batch_num_objs_.clear();
  for (size_t batch = 0; batch < N; ++batch) {
    bool flag = false;
    for (size_t j = 0; j < bottom[3]->count() / bottom[3]->shape(0);) {
      if (label_data[batch * bottom[3]->shape(1) + j] == -1) {
        batch_num_objs_.push_back(j / 5);
        flag = true;
        break;
      }
      j += 5;
    }
    if (!flag) {
      batch_num_objs_.push_back(bottom[3]->shape(1) / 5);
    }
    DCHECK_GE(batch_num_objs_[batch], 0);
  }
  CHECK_EQ(batch_num_objs_.size(), N);

  // Reshape vectors as per the batch size
  final_prob_.resize(N);
  for (size_t batch=0; batch < N; ++batch) {
    final_prob_[batch].resize(anchors_);
    for (size_t anchor = 0; anchor < anchors_; ++anchor) {
      final_prob_[batch][anchor].resize(classes_);
    }
  }
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
  transformed_bbox_preds_.resize(N);
  min_max_pred_bboxs_.resize(N);
  for (int i = 0; i < N; ++i) {
    transformed_bbox_preds_[i].resize(anchors_);
    min_max_pred_bboxs_[i].resize(anchors_);
    for (int j = 0; j < anchors_; ++j) {
      transformed_bbox_preds_[i][j].resize(4);
      min_max_pred_bboxs_[i][j].resize(4);
    }
  }

  CHECK_EQ(bottom[3]->count(), N * 5 * (*std::max_element(batch_num_objs_.begin(),
    batch_num_objs_.end())));
  // Extract bounding box data from batch of images
  for (size_t batch = 0; batch < N; ++batch) {
    for (size_t obj = 0; obj < batch_num_objs_[batch]; ++obj) {
      min_max_gtruth_[batch][obj][0] = bottom[3]->data_at(batch, obj * 5
          + 0, 0, 0);
      min_max_gtruth_[batch][obj][1] = bottom[3]->data_at(batch, obj * 5
          + 1, 0, 0);
      min_max_gtruth_[batch][obj][2] = bottom[3]->data_at(batch, obj * 5
          + 2, 0, 0);
      min_max_gtruth_[batch][obj][3] = bottom[3]->data_at(batch, obj * 5
          + 3, 0, 0);
      min_max_gtruth_[batch][obj][4] = bottom[3]->data_at(batch, obj * 5
          + 4, 0, 0);
    }
  }

  // Transform relative coordinates from ConvDet to bounding box coordinates
  // @f$ [x_{i}^{p}, y_{j}^{p}, h_{k}^{p}, w_{k}^{p}] @f$
  CHECK_EQ(bottom[2]->shape(3), anchors_per_grid * 4);
  transform_bbox_predictions(delta_bbox, &anchors_values_, \
    &transformed_bbox_preds_);

  // Transform each predicted bounding boxes from the form:
  // @f$ [x_{i}^{p}, y_{j}^{p}, h_{k}^{p}, w_{k}^{p}] @f$ to
  // @f$ [xmin, ymin, xmax, ymax] @f$
  transform_bbox(&transformed_bbox_preds_, &min_max_pred_bboxs_);

  // Ensure that `(xmin, ymin, xmax, ymax)` for each of the predicted box
  // are within range of image dimensions
  assert_predictions(&min_max_pred_bboxs_);

  // Transform back the valid `(xmin, ymin, xmax, ymax)` to
  // @f$ [center_x, center_y, anchor_height, anchor_width] @f$
  transform_bbox_inv(&min_max_pred_bboxs_, &transformed_bbox_preds_);

  // Compute IOU for the predicted boxes and the ground truth
  intersection_over_union(&min_max_pred_bboxs_, &min_max_gtruth_, &iou_);

  // Compute the final probability values for each anchor
  for (size_t batch = 0; batch < N; ++batch) {
    for (size_t height = 0; height < H; ++height) {
      for (size_t width = 0; width < W; ++width) {
        for (size_t ch = 0; ch < anchors_per_grid * classes_; ++ch) {
          const int anchor_idx = \
              (height * W + width) * anchors_per_grid + \
              ch / classes_;
          Dtype pr_object = \
              conf_scores[((batch * H + height) * W + \
              width) * anchors_per_grid + ch / classes_];
          Dtype pr_class_idx = \
              class_scores[(batch * anchors_ + \
              anchor_idx) * classes_ + ch % classes_];
          final_prob_[batch][anchor_idx][ch % classes_] = \
              caffe::caffe_cpu_dot(1, &pr_object, &pr_class_idx);
        }
      }
    }
  }

  std::vector<Dtype> class_reg_loss;
  std::vector<Dtype> conf_reg_loss;
  std::vector<Dtype> bbox_reg_loss;
  Dtype conf_loss = 0;
  Dtype class_loss = 0;
  Dtype bbox_loss = 0;
  Dtype tot_loss = 0;

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
      l -= (log(class_scores[(batch * anchors_ + max_iou_indx) * classes_ \
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
      Dtype exp_val, diff_val;
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          for (size_t c = 0; c < anchors_per_grid; ++c) {
            const int anchor_idx = (h * W + w) * anchors_per_grid + c;
            if (anchor_idx == max_iou_indx) {
              diff_val = \
                conf_scores[((batch * H + h) * W + \
                w) * anchors_per_grid + c] - iou_[batch][obj][max_iou_indx];
              exp_val = caffe::caffe_cpu_dot(1, &diff_val, &diff_val);
              l1 += exp_val;
            } else {
              diff_val = conf_scores[((batch * H + h) * W + \
                w) * anchors_per_grid + c];
              exp_val = caffe::caffe_cpu_dot(1, &diff_val, &diff_val);
              l2 += exp_val;
            }
          }
        }
      }
    }
    conf_reg_loss.push_back((float(pos_conf_) / batch_num_objs_[batch]) * l1 \
                          + (float(neg_conf_) / (anchors_ - batch_num_objs_[batch])) * l2);
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
      for (size_t height = 0; height < H; ++height) {
        for (size_t width = 0; width < W; ++width) {
          for (size_t anchor = 0; anchor < anchors_per_grid * 4; ) {
            const int anchor_idx = (height * W + \
              width) * anchors_per_grid + (anchor / 4);
            DCHECK_GE(anchor_idx, 0);
            DCHECK_LT(anchor_idx, anchors_);
            if (anchor_idx == max_iou_indx) {
              Dtype diff_val_x, diff_val_y, diff_val_h, diff_val_w;
              Dtype exp_val_x, exp_val_y, exp_val_h, exp_val_w;
              diff_val_x = \
                delta_bbox[((batch * H + height) * W + \
                width) * anchors_per_grid * 4 + anchor + 0] \
                - gtruth_inv_[batch][obj][anchor_idx][0];
              diff_val_y = \
                delta_bbox[((batch * H + height) * W + \
                width) * anchors_per_grid * 4 + anchor + 1] \
                - gtruth_inv_[batch][obj][anchor_idx][1];
              diff_val_h = \
                delta_bbox[((batch * H + height) * W + \
                width) * anchors_per_grid * 4 + anchor + 2] \
                - gtruth_inv_[batch][obj][anchor_idx][2];
              diff_val_w = \
                delta_bbox[((batch * H + height) * W + \
                width) * anchors_per_grid * 4 + anchor + 3] \
                - gtruth_inv_[batch][obj][anchor_idx][3];
              exp_val_x = caffe::caffe_cpu_dot(1, &diff_val_x, &diff_val_x);
              exp_val_y = caffe::caffe_cpu_dot(1, &diff_val_y, &diff_val_y);
              exp_val_h = caffe::caffe_cpu_dot(1, &diff_val_h, &diff_val_h);
              exp_val_w = caffe::caffe_cpu_dot(1, &diff_val_w, &diff_val_w);
              l += (exp_val_x + exp_val_y + exp_val_h + exp_val_w);
            }
            anchor += 4;
          }
        }
      }
    }
    bbox_reg_loss.push_back((l * lambda_bbox_) / (batch_num_objs_[batch]));
  }

  for (typename std::vector<Dtype>::iterator itr = class_reg_loss.begin();
      itr != class_reg_loss.end(); ++itr) {
    class_loss += (*itr);
  }
  for (typename std::vector<Dtype>::iterator itr = conf_reg_loss.begin();
      itr != conf_reg_loss.end(); ++itr) {
    conf_loss += (*itr);
  }
  for (typename std::vector<Dtype>::iterator itr = bbox_reg_loss.begin();
      itr != bbox_reg_loss.end(); ++itr) {
    bbox_loss += (*itr);
  }
  tot_loss = class_loss + conf_loss + bbox_loss;

  top[0]->mutable_cpu_data()[0] = tot_loss / N;

  if (this->phase_ == caffe::TEST) {
    std::vector<std::vector<Dtype> > batch_filtered_probs(N);
    std::vector<std::vector<size_t> > batch_filtered_idxs(N);
    std::vector<std::vector<std::vector<Dtype> > > batch_filtered_boxes(N);

    for (size_t batch = 0; batch < N; ++batch) {
     filter_predictions(&min_max_pred_bboxs_[batch], &final_prob_[batch],
        &batch_filtered_probs[batch], &batch_filtered_idxs[batch],
        &batch_filtered_boxes[batch], conf_scores, iou_,
        class_scores);
    }
  }
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

        (*iou_)[batch][obj][anchor] = inter_ / (union_ + epsilon);
      }
    }
  }
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::intersection_over_union(
    std::vector<std::vector<Dtype> > *boxes,
    std::vector<Dtype> *base_box, std::vector<float> *iou_) {

  Dtype g_xmin = (*base_box)[0];
  Dtype g_ymin = (*base_box)[1];
  Dtype g_xmax = (*base_box)[2];
  Dtype g_ymax = (*base_box)[3];
  Dtype base_box_w = g_xmax - g_xmin;
  Dtype base_box_h = g_ymax - g_ymin;
  for (size_t b = 0; b < (*boxes).size(); ++b) {
    Dtype xmin = std::max((*boxes)[b][0], g_xmin);
    Dtype ymin = std::max((*boxes)[b][1], g_ymin);
    Dtype xmax = std::min((*boxes)[b][2], g_xmax);
    Dtype ymax = std::min((*boxes)[b][3], g_ymax);

    // Intersection
    Dtype w = std::max(static_cast<Dtype>(0.0), xmax - xmin);
    Dtype h = std::max(static_cast<Dtype>(0.0), ymax - ymin);
    // Union
    Dtype test_box_w = (*boxes)[b][2] - (*boxes)[b][0];
    Dtype test_box_h = (*boxes)[b][3] - (*boxes)[b][1];

    float inter_ = w * h;
    float union_ = test_box_h * test_box_w + base_box_h * base_box_w - inter_;
    (*iou_)[b] = inter_ / (union_ + epsilon);
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
    const Dtype* delta_bbox,
    std::vector<std::vector<Dtype> > *anchors_values_,
    std::vector<std::vector<std::vector<Dtype> > > *transformed_bbox_preds_) {
  for (size_t batch = 0; batch < N; ++batch) {
    for (size_t height = 0; height < H; ++height) {
      for (size_t width = 0; width < W; ++width) {
        for (size_t anchor = 0; anchor < anchors_per_grid * 4;) {
          const int anchor_idx = (height * W + width) * anchors_per_grid + (anchor / 4);
          const Dtype delta_x = \
            delta_bbox[((batch * H + height) * W + \
            width) * anchors_per_grid * 4 + anchor + 0];
          const Dtype delta_y = \
            delta_bbox[((batch * H + height) * W + \
            width) * anchors_per_grid * 4 + anchor + 1];
          const Dtype delta_h = \
            delta_bbox[((batch * H + \
            height) * W + width) * anchors_per_grid * 4 + anchor + 2];
          const Dtype delta_w = \
            delta_bbox[((batch * H + \
            height) * W + width) * anchors_per_grid * 4 + anchor + 3];

          Dtype exp_delta_h, exp_delta_w;
          caffe::caffe_exp(1, &delta_h, &exp_delta_h);
          caffe::caffe_exp(1, &delta_w, &exp_delta_w);

          (*transformed_bbox_preds_)[batch][anchor_idx][0] = \
            (*anchors_values_)[anchor_idx][0] + (*anchors_values_)[anchor_idx][3] * delta_x;
          (*transformed_bbox_preds_)[batch][anchor_idx][1] = \
            (*anchors_values_)[anchor_idx][1] + (*anchors_values_)[anchor_idx][2] * delta_y;
          (*transformed_bbox_preds_)[batch][anchor_idx][2] = \
            (*anchors_values_)[anchor_idx][2] * exp_delta_h;
          (*transformed_bbox_preds_)[batch][anchor_idx][3] = \
            (*anchors_values_)[anchor_idx][3] * exp_delta_w;

          anchor += 4;
        }
      }
    }
  }
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::assert_predictions(
    std::vector<std::vector<std::vector<Dtype> > > *min_max_pred_bboxs_) {
  for (size_t batch = 0; batch < N; ++batch) {
    for (size_t anchor = 0; anchor < anchors_; ++anchor) {
      Dtype p_xmin = (*min_max_pred_bboxs_)[batch][anchor][0];
      Dtype p_ymin = (*min_max_pred_bboxs_)[batch][anchor][1];
      Dtype p_xmax = (*min_max_pred_bboxs_)[batch][anchor][2];
      Dtype p_ymax = (*min_max_pred_bboxs_)[batch][anchor][3];
      (*min_max_pred_bboxs_)[batch][anchor][0] = std::min(std::max(
        static_cast<Dtype>(0.0), p_xmin), image_width -
        static_cast<Dtype>(1.0));
      (*min_max_pred_bboxs_)[batch][anchor][1] = std::min(std::max(
        static_cast<Dtype>(0.0), p_ymin), image_height -
        static_cast<Dtype>(1.0));
      (*min_max_pred_bboxs_)[batch][anchor][2] = std::max(std::min(
        image_width - static_cast<Dtype>(1.0), p_xmax),
        static_cast<Dtype>(0.0));
      (*min_max_pred_bboxs_)[batch][anchor][3] = std::max(std::min(
        image_height - static_cast<Dtype>(1.0), p_ymax),
        static_cast<Dtype>(0.0));
    }
  }
}

template <typename Dtype>
std::vector<bool> SqueezeDetLossLayer<Dtype>::non_maximal_suppression(
    std::vector<std::vector<Dtype> > *boxes, std::vector<Dtype> *probs) {
  std::vector<bool> keep;
  if ((*boxes).size() == 0) {
    keep.resize(0);
  } else {
    keep.resize((*boxes).size());
    std::fill(keep.begin(), keep.end(), true);
    std::vector<size_t> prob_args_sorted((*probs).size());
    std::iota(prob_args_sorted.begin(), prob_args_sorted.end(), 0);
    std::sort(prob_args_sorted.begin(), prob_args_sorted.end(), \
      [probs](size_t i1, size_t i2) {return (*probs)[i1] > (*probs)[i2];});

    for (std::vector<size_t>::iterator itr = prob_args_sorted.begin();
        itr != prob_args_sorted.end()-1; ++itr) {
      const int idx = itr - prob_args_sorted.begin();
      std::vector<float> iou_(prob_args_sorted.size()-idx-1);
      std::vector<std::vector<Dtype> > temp_boxes(iou_.size());
      for (size_t bb = 0; bb < temp_boxes.size(); ++bb) {
        std::vector<Dtype> temp_box(4);
        for (size_t b = 0; b < 4; ++b) {
          temp_box[b] = (*boxes)[prob_args_sorted[idx+bb+1]][b];
        }
        temp_boxes[bb] = temp_box;
      }
      intersection_over_union(&temp_boxes, \
          &(*boxes)[prob_args_sorted[idx]], &iou_);
      for (std::vector<float>::iterator _itr = iou_.begin();
          _itr != iou_.end(); ++_itr) {
        const int iou_idx = _itr - iou_.begin();
        if (*_itr > intersection_thresh) {
          keep[prob_args_sorted[idx+iou_idx+1]] = false;
        }
      }
    }
  }
  return keep;
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::filter_predictions(std::vector<std::vector<
    Dtype> > *boxes, std::vector<std::vector<Dtype> > *probs,
    std::vector<Dtype> *filtered_probs, std::vector<size_t> *filtered_idxs,
    std::vector<std::vector<Dtype> > *filtered_boxes, const Dtype* conf_scores,
    std::vector<std::vector<std::vector<float> > > iou_,
    const Dtype* class_scores) {

  std::vector<Dtype> max_class_probs((*probs).size());
  std::vector<size_t> args((*probs).size());
  std::iota(args.begin(), args.end(), 0);
  CHECK_EQ((*probs).size(), anchors_);
  CHECK_EQ((*boxes).size(), anchors_);
  for (size_t box = 0; box < (*boxes).size(); ++box) {
    const int _prob_idx = \
        std::max_element((*probs)[box].begin(),
        (*probs)[box].end()) - (*probs)[box].begin();
    max_class_probs[box] = (*probs)[box][_prob_idx];
  }
  std::sort(args.begin(), args.end(), \
    [&max_class_probs](size_t i1, size_t i2) \
    {return max_class_probs[i1] > max_class_probs[i2];});
  std::vector<size_t> top_n_order(args.begin(), \
    args.begin()+n_top_detections);

  std::vector<std::vector<Dtype> > top_n_boxes(n_top_detections);
  std::vector<size_t> top_n_idxs(n_top_detections);
  std::vector<Dtype> top_n_probs(n_top_detections);
  for (size_t i = 0; i < n_top_detections; ++i) {
    top_n_boxes[i].resize(4);
  }

  std::vector<Dtype> dummy_probability;
  for (size_t b=0; b<N; ++b) {
    for (size_t h=0; h<H; ++h) {
      for (size_t w=0; w<W; ++w) {
        for (size_t k=0; k<9; ++k) {
          const int anch = ((b*H+h)*W+w)*9+k;
          DCHECK_LT(anch, anchors_);
          Dtype _prob_ = conf_scores[anch];
          dummy_probability.push_back(_prob_);
        }
      }
    }
  }
  for (size_t n = 0; n < n_top_detections; ++n) {
    top_n_probs[n] = max_class_probs[top_n_order[n]];
    top_n_idxs[n]  = \
        std::max_element((*probs)[top_n_order[n]].begin(), \
        (*probs)[top_n_order[n]].end()) - \
        (*probs)[top_n_order[n]].begin();
    for (size_t i = 0; i < 4; ++i) {
      top_n_boxes[n][i] = (*boxes)[top_n_order[n]][i];
    }

    if (top_n_probs[n] > intersection_thresh) {
      LOG(INFO) << n << ": " << top_n_probs[n] << " "
                << top_n_idxs[n] << " "
                << top_n_boxes[n][0] << " " << top_n_boxes[n][1] << " "
                << top_n_boxes[n][2] << " " << top_n_boxes[n][3] << "\n";
    }
  }
  return ;

  for (size_t c = 0; c < classes_; ++c) {
    std::vector<size_t> idxs_per_class;
    for (size_t n = 0; n < n_top_detections; ++n) {
      if (top_n_idxs[n] == c) {
        idxs_per_class.push_back(n);
      }
    }
    std::vector<std::vector<Dtype> > boxes_per_class(idxs_per_class.size());
    std::vector<Dtype> probs_per_class(idxs_per_class.size());
    std::vector<bool> keep_per_class;
    for (std::vector<size_t>::iterator itr = idxs_per_class.begin();
        itr != idxs_per_class.end(); ++itr) {
      const int idx = itr - idxs_per_class.begin();
      probs_per_class[idx] = top_n_probs[*itr];
      for (size_t b = 0; b < 4; ++b) {
        boxes_per_class[idx].push_back(top_n_boxes[*itr][b]);
      }
    }
    keep_per_class = \
        non_maximal_suppression(&boxes_per_class, &probs_per_class);
    for (std::vector<bool>::iterator itr = keep_per_class.begin();
        itr != keep_per_class.end(); ++itr) {
      const int idx = itr - keep_per_class.begin();
      if (*itr) {
        (*filtered_idxs).push_back(c);
        (*filtered_probs).push_back(probs_per_class[idx]);
        std::vector<Dtype> temp_box(4);
        for (size_t b = 0; b < 4; ++b) {
          temp_box[b] = boxes_per_class[idx][b];
        }
        (*filtered_boxes).push_back(temp_box);
      }
    }
  }
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[3]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0] && propagate_down[1] && propagate_down[2]) {
    Dtype* bottom_class_diff = bottom[0]->mutable_cpu_diff();
    Dtype* bottom_conf_diff  = bottom[1]->mutable_cpu_diff();
    Dtype* bottom_bbox_diff  = bottom[2]->mutable_cpu_diff();

    const Dtype* class_scores = bottom[0]->cpu_data();
    const Dtype* conf_scores  = bottom[1]->cpu_data();
    const Dtype* delta_bbox   = bottom[2]->cpu_data();

    std::vector<std::vector<std::vector<std::vector<Dtype> > > > gtruth_inv_;
    std::vector<std::vector<std::vector<Dtype> > > gtruth_, min_max_gtruth_;
    std::vector<std::vector<std::vector<Dtype> > > transformed_bbox_preds_;
    std::vector<std::vector<std::vector<Dtype> > > min_max_pred_bboxs_;
    std::vector<std::vector<std::vector<float> > > iou_;

    min_max_gtruth_.resize(N);
    gtruth_.resize(N);
    iou_.resize(N);
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
    transformed_bbox_preds_.resize(N);
    min_max_pred_bboxs_.resize(N);
    for (int i = 0; i < N; ++i) {
      transformed_bbox_preds_[i].resize(anchors_);
      min_max_pred_bboxs_[i].resize(anchors_);
      for (int j = 0; j < anchors_; ++j) {
        transformed_bbox_preds_[i][j].resize(4);
        min_max_pred_bboxs_[i][j].resize(4);
      }
    }
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

    // Extract bounding box data from batch of images
    for (size_t batch = 0; batch < N; ++batch) {
      for (size_t obj = 0; obj < batch_num_objs_[batch]; ++obj) {
        min_max_gtruth_[batch][obj][0] = bottom[3]->data_at(batch, obj * 5
            + 0, 0, 0);
        min_max_gtruth_[batch][obj][1] = bottom[3]->data_at(batch, obj * 5
            + 1, 0, 0);
        min_max_gtruth_[batch][obj][2] = bottom[3]->data_at(batch, obj * 5
            + 2, 0, 0);
        min_max_gtruth_[batch][obj][3] = bottom[3]->data_at(batch, obj * 5
            + 3, 0, 0);
        min_max_gtruth_[batch][obj][4] = bottom[3]->data_at(batch, obj * 5
            + 4, 0, 0);
      }
    }

    CHECK_EQ(bottom[2]->shape(3), anchors_per_grid * 4);
    transform_bbox_predictions(delta_bbox, &anchors_values_, \
      &transformed_bbox_preds_);
    // Transform each predicted bounding boxes from the form:
    // @f$ [x_{i}^{p}, y_{j}^{p}, h_{k}^{p}, w_{k}^{p}] @f$ to
    // @f$ [xmin, ymin, xmax, ymax] @f$
    transform_bbox(&transformed_bbox_preds_, &min_max_pred_bboxs_);

    // Ensure that `(xmin, ymin, xmax, ymax)` for each of the predicted box
    // are within range of image dimensions
    assert_predictions(&min_max_pred_bboxs_);

    // Compute IOU for the predicted boxes and the ground truth
    intersection_over_union(&min_max_pred_bboxs_, &min_max_gtruth_, &iou_);
    transform_bbox_inv(&min_max_gtruth_, &gtruth_);
    gtruth_inv_transform(&gtruth_, &batch_num_objs_, &gtruth_inv_);

    // GRADIENTS FOR CROSS ENTROPY LOSS
    for (size_t batch = 0; batch < N; ++batch) {
      for (size_t obj = 0; obj < batch_num_objs_[batch]; ++obj) {
        const int max_iou_indx = std::max_element(iou_[batch][obj].begin(),
          iou_[batch][obj].end()) - iou_[batch][obj].begin();
        Dtype class_idx = min_max_gtruth_[batch][obj][4];
        DCHECK_GE(max_iou_indx, 0);
        DCHECK_LT(max_iou_indx, anchors_);
        for (size_t anchor = 0; anchor < anchors_; ++anchor) {
          for (size_t id = 0; id < classes_; ++id) {
            Dtype _grad = 0;
            if (anchor == max_iou_indx) {
              const int label_value = static_cast<int>(class_idx);
              DCHECK_GE(label_value, 0);
              DCHECK_LT(label_value, classes_);
              if (label_value == id) {
                _grad = \
                  -1.0 / class_scores[(batch * anchors_ + \
                  anchor) * classes_ + id];
              } else {
                _grad = 0;
              }
            } else {
              _grad = 0;
            }
            if (obj == 0) {
              bottom_class_diff[(batch * anchors_ + \
                anchor) * classes_ + id] = _grad;
            } else {
              bottom_class_diff[(batch * anchors_ + \
                anchor) * classes_ + id] += _grad;
            }
          }
        }
      }
    }
    // GRADIENTS FOR CLASS REGRESSION LOSS
    for (size_t batch = 0; batch < N; ++batch) {
      for (size_t obj = 0; obj < batch_num_objs_[batch]; ++obj) {
        const int max_iou_indx = std::max_element(iou_[batch][obj].begin(),
          iou_[batch][obj].end()) - iou_[batch][obj].begin();
        for (size_t height = 0; height < H; ++height) {
          for (size_t width = 0; width < W; ++width) {
            for (size_t anchor = 0; anchor < anchors_per_grid; ++anchor) {
              Dtype _grad = 0;
              const int anchor_idx = \
                (height * W + width) * anchors_per_grid + anchor;
              Dtype conf_score_truth = \
                iou_[batch][obj][anchor_idx];
              Dtype conf_score_pred  = \
                conf_scores[((batch * H + height) * W + \
                width) * anchors_per_grid + anchor];
              DCHECK_GE(anchor_idx, 0);
              DCHECK_LT(anchor_idx, anchors_);
              if (anchor_idx == max_iou_indx) {
                _grad = ((2.0 * pos_conf_) / (batch_num_objs_[batch])) * \
                        (conf_score_pred - conf_score_truth);
              } else {
                _grad =  ((2.0 * neg_conf_) / (anchors_ - \
                        batch_num_objs_[batch])) * conf_score_pred;
              }
              if (obj == 0) {
                bottom_conf_diff[((batch * H + height) * W + \
                  width) * anchors_per_grid + anchor] = _grad;
              } else {
                bottom_conf_diff[((batch * H + height) * W + \
                  width) * anchors_per_grid + anchor] += _grad;
              }
            }
          }
        }
      }
    }
    // GRADIENTS FOR BBOX REGRESSION LOSS
    for (size_t batch = 0; batch < N; ++batch) {
      for (size_t obj = 0; obj < batch_num_objs_[batch]; ++obj) {
        const int max_iou_indx = std::max_element(iou_[batch][obj].begin(),
          iou_[batch][obj].end()) - iou_[batch][obj].begin();
        DCHECK_GE(max_iou_indx, 0);
        DCHECK_LT(max_iou_indx, anchors_);
        for (size_t height = 0; height < H; ++height) {
          for (size_t width = 0; width < W; ++width) {
            for (size_t id = 0; id < anchors_per_grid * 4;) {
              const int anchor_idx = \
                (height * W + width) * anchors_per_grid + (id / 4);
              DCHECK_GE(anchor_idx, 0);
              DCHECK_LT(anchor_idx, anchors_);
              Dtype _grad_x, _grad_y, _grad_w, _grad_h;
              if (anchor_idx == max_iou_indx) {
                Dtype delta_pred, delta_truth;
                delta_pred = \
                  delta_bbox[((batch * H + height) * W + \
                  width) * anchors_per_grid * 4 + id + 0];
                delta_truth = \
                  gtruth_inv_[batch][obj][anchor_idx][0];
                _grad_x = ((2 * lambda_bbox_) / batch_num_objs_[batch]) * \
                  (delta_pred - delta_truth);
                delta_pred = \
                  delta_bbox[((batch * H + height) * W + \
                  width) * anchors_per_grid * 4 + id + 1];
                delta_truth = \
                  gtruth_inv_[batch][obj][anchor_idx][1];
                _grad_y = ((2 * lambda_bbox_) / batch_num_objs_[batch]) * \
                  (delta_pred - delta_truth);
                delta_pred = \
                  delta_bbox[((batch * H + height) * W + \
                  width) * anchors_per_grid * 4 + id + 2];
                delta_truth = \
                  gtruth_inv_[batch][obj][anchor_idx][2];
                _grad_h = ((2 * lambda_bbox_) / batch_num_objs_[batch]) * \
                  (delta_pred - delta_truth);
                delta_pred = \
                  delta_bbox[((batch * H + height) * W + \
                  width) * anchors_per_grid * 4 + id + 3];
                delta_truth = \
                  gtruth_inv_[batch][obj][anchor_idx][3];
                _grad_w = ((2 * lambda_bbox_) / batch_num_objs_[batch]) * \
                  (delta_pred - delta_truth);
              } else {
                _grad_x = 0;
                _grad_y = 0;
                _grad_h = 0;
                _grad_w = 0;
              }
              if (obj == 0) {
                bottom_bbox_diff[((batch * H + height) * W + \
                   width) * anchors_per_grid * 4 + id + 0] = _grad_x;
                bottom_bbox_diff[((batch * H + height) * W + \
                   width) * anchors_per_grid * 4 + id + 1] = _grad_y;
                bottom_bbox_diff[((batch * H + height) * W + \
                   width) * anchors_per_grid * 4 + id + 2] = _grad_h;
                bottom_bbox_diff[((batch * H + height) * W + \
                   width) * anchors_per_grid * 4 + id + 3] = _grad_w;
              } else {
                bottom_bbox_diff[((batch * H + height) * W + \
                   width) * anchors_per_grid * 4 + id + 0] += _grad_x;
                bottom_bbox_diff[((batch * H + height) * W + \
                   width) * anchors_per_grid * 4 + id + 1] += _grad_y;
                bottom_bbox_diff[((batch * H + height) * W + \
                   width) * anchors_per_grid * 4 + id + 2] += _grad_h;
                bottom_bbox_diff[((batch * H + height) * W + \
                   width) * anchors_per_grid * 4 + id + 3] += _grad_w;
              }
               id += 4;
            }
          }
        }
      }
    }
    // TODO @kvmanohar : Use normalization parameter from protobuf
    Dtype loss_weight = top[0]->cpu_diff()[0] / N;
    caffe::caffe_scal(bottom[0]->count(), loss_weight, bottom_class_diff);
    caffe::caffe_scal(bottom[1]->count(), loss_weight, bottom_conf_diff);
    caffe::caffe_scal(bottom[2]->count(), loss_weight, bottom_bbox_diff);
  } else {
    LOG(FATAL) << "Should backpropagate to all the input blobs except labels.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(SqueezeDetLossLayer);
#endif

INSTANTIATE_CLASS(SqueezeDetLossLayer);
REGISTER_LAYER_CLASS(SqueezeDetLoss);

}  // namespace caffe
