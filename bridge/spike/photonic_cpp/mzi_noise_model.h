#ifndef PHOTONIC_CPP_MZI_NOISE_MODEL_H
#define PHOTONIC_CPP_MZI_NOISE_MODEL_H

#include "photonic_model.h"
#include "noise_components.h"
#include "config.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace photonic {

/**
 * MZI Realistic Noise Model.
 *
 * Implements a complete photonic MZI (Mach-Zehnder Interferometer) model
 * with realistic noise and non-idealities:
 *
 * 1. DAC Quantization (input)
 * 2. Phase Error
 * 3. Thermal Crosstalk
 * 4. Ideal MVM computation
 * 5. Insertion Loss
 * 6. Output Crosstalk
 * 7. Detector Noise
 * 8. ADC Quantization (output)
 *
 * Based on: Liu et al., "FIONA: Photonic-Electronic Co-Simulation Framework",
 * ICCAD 2023.
 */
class MZINoiseModel : public PhotonicModel {
public:
    explicit MZINoiseModel(const PhotonicConfig& config)
        : config_(config), rng_(config.random_seed) {
        if (config_.debug) {
            std::cout << "[MZINoiseModel] Initialized with:" << std::endl;
            config_.print();
        }
    }

    void mvm(int16_t* result, const int16_t* matrix,
             const int16_t* vector, size_t rows, size_t cols) override {
        // Convert to float for noise processing
        std::vector<float> mat_f(rows * cols);
        std::vector<float> vec_f(cols);
        std::vector<float> res_f(rows);

        for (size_t i = 0; i < rows * cols; ++i) {
            mat_f[i] = static_cast<float>(matrix[i]);
        }
        for (size_t i = 0; i < cols; ++i) {
            vec_f[i] = static_cast<float>(vector[i]);
        }

        // Apply forward pass with all noise components
        forward(res_f.data(), mat_f.data(), vec_f.data(), rows, cols);

        // Convert back to int16
        for (size_t i = 0; i < rows; ++i) {
            float val = std::round(res_f[i]);
            if (val > 32767.0f) val = 32767.0f;
            if (val < -32768.0f) val = -32768.0f;
            result[i] = static_cast<int16_t>(val);
        }
    }

    int16_t dotp(const int16_t* vec1, const int16_t* vec2, size_t len) override {
        // Treat DOTP as 1-row MVM
        std::vector<float> mat_f(len);
        std::vector<float> vec_f(len);
        float result_f;

        for (size_t i = 0; i < len; ++i) {
            mat_f[i] = static_cast<float>(vec1[i]);
            vec_f[i] = static_cast<float>(vec2[i]);
        }

        forward(&result_f, mat_f.data(), vec_f.data(), 1, len);

        float val = std::round(result_f);
        if (val > 32767.0f) val = 32767.0f;
        if (val < -32768.0f) val = -32768.0f;
        return static_cast<int16_t>(val);
    }

    std::string name() const override { return "mzi_realistic"; }

private:
    PhotonicConfig config_;
    noise::RandomGenerator rng_;

    /**
     * Forward pass with all noise components.
     */
    void forward(float* result, float* matrix, float* vector,
                 size_t rows, size_t cols) {
        // 1. DAC Quantization (input)
        if (config_.quant_bits > 0) {
            noise::quantize(vector, cols, config_.quant_bits, true);
        }

        // 2. Phase Error on weights
        if (config_.phase_error_sigma > 0.0f) {
            noise::apply_phase_error(matrix, rows * cols,
                                     config_.phase_error_sigma, rng_);
        }

        // 3. Thermal Crosstalk
        if (config_.thermal_crosstalk_sigma > 0.0f) {
            noise::apply_thermal_crosstalk(matrix, rows, cols,
                                           config_.thermal_crosstalk_sigma, rng_);
        }

        // 4. Ideal MVM computation
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            mat(matrix, rows, cols);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>
            vec(vector, cols);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>
            res(result, rows);

        res = mat * vec;

        // 5. Insertion Loss
        if (config_.insertion_loss_db > 0.0f && config_.num_mzi_stages > 0) {
            noise::apply_insertion_loss(result, rows,
                                        config_.insertion_loss_db,
                                        config_.num_mzi_stages);
        }

        // 6. Output Crosstalk
        if (config_.crosstalk_db < 0.0f) {
            noise::apply_output_crosstalk(result, rows, config_.crosstalk_db);
        }

        // 7. Detector Noise
        if (config_.detector_noise_sigma > 0.0f) {
            noise::apply_detector_noise(result, rows,
                                        config_.detector_noise_sigma, rng_);
        }

        // 8. ADC Quantization (output)
        if (config_.quant_bits > 0) {
            noise::quantize(result, rows, config_.quant_bits, false);
        }
    }
};

/**
 * Quantized Model (DAC/ADC only, no other noise).
 */
class QuantizedModel : public PhotonicModel {
public:
    explicit QuantizedModel(const PhotonicConfig& config)
        : config_(config) {}

    void mvm(int16_t* result, const int16_t* matrix,
             const int16_t* vector, size_t rows, size_t cols) override {
        // Convert to float
        std::vector<float> mat_f(rows * cols);
        std::vector<float> vec_f(cols);
        std::vector<float> res_f(rows);

        for (size_t i = 0; i < rows * cols; ++i) {
            mat_f[i] = static_cast<float>(matrix[i]);
        }
        for (size_t i = 0; i < cols; ++i) {
            vec_f[i] = static_cast<float>(vector[i]);
        }

        // DAC quantization
        if (config_.quant_bits > 0) {
            noise::quantize(vec_f.data(), cols, config_.quant_bits, true);
        }

        // Ideal MVM
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            mat(mat_f.data(), rows, cols);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>
            vec(vec_f.data(), cols);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>
            res(res_f.data(), rows);

        res = mat * vec;

        // ADC quantization
        if (config_.quant_bits > 0) {
            noise::quantize(res_f.data(), rows, config_.quant_bits, false);
        }

        // Convert back to int16
        for (size_t i = 0; i < rows; ++i) {
            float val = std::round(res_f[i]);
            if (val > 32767.0f) val = 32767.0f;
            if (val < -32768.0f) val = -32768.0f;
            result[i] = static_cast<int16_t>(val);
        }
    }

    int16_t dotp(const int16_t* vec1, const int16_t* vec2, size_t len) override {
        std::vector<int16_t> result(1);
        mvm(result.data(), vec1, vec2, 1, len);
        return result[0];
    }

    std::string name() const override { return "quantized"; }

private:
    PhotonicConfig config_;
};

} // namespace photonic

#endif // PHOTONIC_CPP_MZI_NOISE_MODEL_H
