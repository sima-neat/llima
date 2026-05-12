//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2025 SiMa.ai, All Rights Reserved.  ||
//**************************************************************************
// NOTICE:  All information contained herein is, and remains the property of
// SiMa.ai. The intellectual and technical concepts contained herein are
// proprietary to SiMa and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// Dissemination of this information or reproduction of this material is
// strictly forbidden unless prior written permission is obtained from
// SiMa.ai.  Access to the source code contained herein is hereby forbidden
// to anyone except current SiMa.ai employees, managers or contractors who
// have executed Confidentiality and Non-disclosure agreements explicitly
// covering such access.
//
// The copyright notice above does not evidence any actual or intended
// publication or disclosure  of  this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
//
// ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
// DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
// CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
// LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
// CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
// REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
// SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
//
//**************************************************************************

#ifndef _SIMA_LLIMA_EIGEN_TYPES_
#define _SIMA_LLIMA_EIGEN_TYPES_

#define EIGEN_USE_THREADS 1
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


using ArrayXXcd = Eigen::Array<
    std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
>;
using ArrayXXd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXXbf = Eigen::Array<Eigen::bfloat16, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXd = Eigen::Array<double, 1, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXf = Eigen::Array<float, 1, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXi16 = Eigen::Array<int16_t, 1, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXbf = Eigen::Array<Eigen::bfloat16, 1, Eigen::Dynamic, Eigen::RowMajor>;

using MatrixXXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXXbf = Eigen::Matrix<
    Eigen::bfloat16, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor
>;

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
