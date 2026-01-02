#ifndef PHOTONIC_CPP_IDEAL_MODEL_H
#define PHOTONIC_CPP_IDEAL_MODEL_H

#include "photonic_model.h"
#include <Eigen/Dense>

namespace photonic {

/**
 * Ideal photonic model.
 *
 * Performs exact matrix-vector multiplication without any noise or
 * non-idealities. This is the fastest model and serves as a baseline.
 */
class IdealModel : public PhotonicModel {
public:
    IdealModel() = default;

    void mvm(int16_t* result, const int16_t* matrix,
             const int16_t* vector, size_t rows, size_t cols) override {
        // Use Eigen for efficient matrix-vector multiplication
        Eigen::Map<const Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            mat(matrix, rows, cols);
        Eigen::Map<const Eigen::Matrix<int16_t, Eigen::Dynamic, 1>>
            vec(vector, cols);
        Eigen::Map<Eigen::Matrix<int16_t, Eigen::Dynamic, 1>>
            res(result, rows);

        // Compute: result = matrix * vector
        // Use int32 for intermediate computation to avoid overflow
        Eigen::Matrix<int32_t, Eigen::Dynamic, 1> temp =
            mat.cast<int32_t>() * vec.cast<int32_t>();

        // Clamp to int16 range and store
        for (size_t i = 0; i < rows; ++i) {
            int32_t val = temp(i);
            if (val > 32767) val = 32767;
            if (val < -32768) val = -32768;
            res(i) = static_cast<int16_t>(val);
        }
    }

    int16_t dotp(const int16_t* vec1, const int16_t* vec2, size_t len) override {
        Eigen::Map<const Eigen::Matrix<int16_t, Eigen::Dynamic, 1>> v1(vec1, len);
        Eigen::Map<const Eigen::Matrix<int16_t, Eigen::Dynamic, 1>> v2(vec2, len);

        // Use int32 for accumulation to avoid overflow
        int32_t result = v1.cast<int32_t>().dot(v2.cast<int32_t>());

        // Clamp to int16 range
        if (result > 32767) result = 32767;
        if (result < -32768) result = -32768;

        return static_cast<int16_t>(result);
    }

    std::string name() const override { return "ideal"; }
    bool is_ideal() const override { return true; }
};

/**
 * Ideal model for float operations.
 */
class IdealModelFloat : public PhotonicModelFloat {
public:
    IdealModelFloat() = default;

    void mvm(float* result, const float* matrix,
             const float* vector, size_t rows, size_t cols) override {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            mat(matrix, rows, cols);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>>
            vec(vector, cols);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>
            res(result, rows);

        res = mat * vec;
    }

    float dotp(const float* vec1, const float* vec2, size_t len) override {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>> v1(vec1, len);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>> v2(vec2, len);
        return v1.dot(v2);
    }

    std::string name() const override { return "ideal_float"; }
    bool is_ideal() const override { return true; }
};

} // namespace photonic

#endif // PHOTONIC_CPP_IDEAL_MODEL_H
