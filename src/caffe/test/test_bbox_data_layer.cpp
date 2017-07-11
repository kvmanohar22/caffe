#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/bbox_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class BboxDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BboxDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);

    MakeTempFilename(&filename_new);
    std::ofstream new_file(filename_new.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_new;
    new_file << EXAMPLES_SOURCE_DIR "images/cat.jpg" << " "
                << EXAMPLES_SOURCE_DIR "annotations/cat.txt"
                << std::endl
                << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg" << " "
                << EXAMPLES_SOURCE_DIR "annotations/fish-bike.txt"
                << std::endl
                << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg" << " "
                << EXAMPLES_SOURCE_DIR "annotations/people.txt"
                << std::endl;
    new_file.close();

  }

  virtual ~BboxDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_new;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BboxDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(BboxDataLayerTest, TestNew) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter param;
    BboxDataParameter* bbox_data_param = param.mutable_bbox_data_param();
    bbox_data_param->set_batch_size(3);
    bbox_data_param->set_source(this->filename_new.c_str());
    bbox_data_param->set_shuffle(false);
    bbox_data_param->set_new_height(416);
    bbox_data_param->set_new_width(416);
    BboxDataLayer<Dtype> layer(param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_EQ(this->blob_top_data_->num(), 3);
    EXPECT_EQ(this->blob_top_data_->channels(), 3);
    EXPECT_EQ(this->blob_top_data_->height(), 416);
    EXPECT_EQ(this->blob_top_data_->width(), 416);
    EXPECT_EQ(this->blob_top_label_->num(), 3);
    EXPECT_EQ(this->blob_top_label_->channels(), 15);
    int arr[3 * 15] = {23, 56, 278, 411, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                       22, 45, 153, 221, 21, 32, 155, 421, 301, 22, -1, -1, -1, -1, -1,
                       23, 56, 278, 411, 0, 22, 45, 153, 221, 1, 32, 155, 421, 301, 2};
    for (size_t i = 0; i < this->blob_top_label_->count(); ++i) {
      EXPECT_EQ(this->blob_top_label_->cpu_data()[i], arr[i]);
    }
}

}  // namespace caffe
#endif  // USE_OPENCV
