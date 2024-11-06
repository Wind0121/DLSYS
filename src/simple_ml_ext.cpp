#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <cstring>

namespace py = pybind11;

size_t pos(size_t i, size_t j, size_t dim0, size_t dim1) {
    return i * dim1 + j;
}

void matmul(const float *X, const float *Y, float *Z, size_t n, size_t k, size_t m) {
    // X: n * k(input)
    // Y: k * m(input)
    // Z: n * m(output)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            Z[pos(i, j, n, m)] = 0;
            for (size_t t = 0; t < k; t++) {
                Z[pos(i, j, n, m)] += X[pos(i, t, n, k)] * Y[pos(t, j, k, m)];
            }
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t iterations = (m + batch - 1) / batch;
    for (size_t iter = 0; iter < iterations; iter++) {
        size_t dim0 = std::min(batch, m - iter * batch);
        const float *xx = &X[pos(iter * batch, 0, m, n)];
        const unsigned char *yy = &y[iter * batch];

        float *z = new float[dim0 * k];
        matmul(xx, theta, z, dim0, n, k);
        for (size_t i = 0; i < dim0 * k; i++) {
            z[i] = std::exp(z[i]);
        }

        for (size_t i = 0; i < dim0; i++) {
            float sum = 0;
            for (size_t j = 0; j < k; j++) {
                sum += z[pos(i, j, dim0, k)];
            }
            for (size_t j = 0; j < k; j++) {
                z[pos(i, j, dim0, k)] /= sum;
            }
        }

        for (size_t i = 0; i < dim0; i++) {
            z[pos(i, yy[i], dim0, k)] -= 1;
        }

        float *xt = new float[n * dim0];
        for (size_t i = 0; i < dim0; i++) {
            for (size_t j = 0; j < n; j++) {
                xt[pos(j, i, n, dim0)] = xx[pos(i, j, dim0, n)];
            }
        }

        float *grad = new float[n * k];
        matmul(xt, z, grad, n, dim0, k);

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < k; j++) {
                theta[pos(i, j, n, k)] -= (lr * grad[pos(i, j, n, k)] / dim0); 
            }
        }

        delete[] z;
        delete[] xt;
        delete[] grad;
    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
