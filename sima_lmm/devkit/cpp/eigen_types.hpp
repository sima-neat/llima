#ifndef _SIMA_LLIMA_EIGEN_TYPES_
#define _SIMA_LLIMA_EIGEN_TYPES_

#define EIGEN_USE_THREADS 1
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using ArrayXXcd =
    Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXXd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXXbf = Eigen::Array<Eigen::bfloat16, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXd = Eigen::Array<double, 1, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXf = Eigen::Array<float, 1, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXi16 = Eigen::Array<int16_t, 1, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXbf = Eigen::Array<Eigen::bfloat16, 1, Eigen::Dynamic, Eigen::RowMajor>;

using MatrixXXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXXbf = Eigen::Matrix<Eigen::bfloat16, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using Tensor3Df = Eigen::Tensor<float, 3, Eigen::RowMajor>;
using Tensor3Dbf = Eigen::Tensor<Eigen::bfloat16, 3, Eigen::RowMajor>;
using Tensor5Dbf = Eigen::Tensor<Eigen::bfloat16, 5, Eigen::RowMajor>;
using Tensor9Dbf = Eigen::Tensor<Eigen::bfloat16, 9, Eigen::RowMajor>;

inline Eigen::ThreadPoolDevice& get_eigen_device() {
  static int num_threads = std::thread::hardware_concurrency();
  static Eigen::ThreadPool pool(num_threads);
  static Eigen::ThreadPoolDevice device(&pool, num_threads);
  return device;
}

#endif
