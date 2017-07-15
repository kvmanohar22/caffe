#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/squeezedet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SqueezeDetlossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
__global__ void SqueezedetlossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label, Dtype* counts) {

            std::fstream file;
            file.open("/users/TeamVideoSummarization/gsoc/dev/caffe/obj_debug_gpu.log", std::fstream::app);
            file << "here" << std::endl;
            file.close();

}

template <typename Dtype>
void SqueezeDetLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(SqueezeDetLossLayer);

}  // namespace caffe
