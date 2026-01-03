#ifndef PHOTONIC_CPP_PCM_NOISE_MODEL_H
#define PHOTONIC_CPP_PCM_NOISE_MODEL_H

#include "photonic_model.h"
#include "noise_components.h"
#include "config.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace photonic {

/**
 * PCM (Phase Change Memory) Photonic Noise Model.
 *
 * Implements a photonic accelerator model based on PCM cells for weight storage.
 * PCM uses phase change materials (e.g., Ge2Sb2Te5) to store analog weights
 * through reflectivity/transmissivity modulation.
 *
 * Noise sources modeled:
 * 1. Programming Variability - Variation in written weight values
 * 2. Resistance Drift - Temporal drift of programmed states (time-dependent)
 * 3. Read Noise - Shot noise and thermal noise during optical readout
 * 4. DAC/ADC Quantization - Limited bit precision
 *
 * References:
 * - Feldmann et al., "All-optical spiking neurosynaptic networks", Nature 2019
 * - Rios et al., "In-memory computing on a photonic platform", Science Advances 2019
 */
class PCMNoiseModel : public PhotonicModel {
public:
    explicit PCMNoiseModel(const PhotonicConfig& config)
        : config_(config), rng_(config.random_seed) {
        if (config_.debug) {
            std::cout << "[PCMNoiseModel] Initialized with:" << std::endl;
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

        // Apply forward pass with PCM noise
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

    std::string name() const override { return "pcm_realistic"; }

private:
    PhotonicConfig config_;
    noise::RandomGenerator rng_;

    /**
     * Forward pass with PCM noise model.
     */
    void forward(float* result, float* matrix, float* vector,
                 size_t rows, size_t cols) {
        // 1. DAC Quantization (input)
        if (config_.quant_bits > 0) {
            noise::quantize(vector, cols, config_.quant_bits, true);
        }

        // 2. Programming Variability on weights
        // PCM programming has inherent variability (~1-5% coefficient of variation)
        if (config_.pcm_prog_variability > 0.0f) {
            apply_programming_variability(matrix, rows * cols);
        }

        // 3. Resistance Drift (simulated as multiplicative noise)
        // PCM states drift over time: R(t) = R0 * (t/t0)^drift_coefficient
        if (config_.pcm_drift_coefficient > 0.0f) {
            apply_resistance_drift(matrix, rows * cols);
        }

        // 4. Ideal MVM computation
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            mat(matrix, rows, cols);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>
            vec(vector, cols);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>
            res(result, rows);

        res = mat * vec;

        // 5. Read Noise (optical readout noise)
        if (config_.pcm_read_noise_sigma > 0.0f) {
            apply_read_noise(result, rows);
        }

        // 6. ADC Quantization (output)
        if (config_.quant_bits > 0) {
            noise::quantize(result, rows, config_.quant_bits, false);
        }
    }

    /**
     * Programming Variability.
     * Each PCM cell has variability when programmed to a target state.
     * Models as multiplicative Gaussian noise: W' = W * (1 + epsilon)
     * where epsilon ~ N(0, sigma^2)
     */
    void apply_programming_variability(float* weights, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            float epsilon = rng_.normal(0.0f, config_.pcm_prog_variability);
            weights[i] *= (1.0f + epsilon);
        }
    }

    /**
     * Resistance Drift.
     * PCM resistance drifts over time following: R(t) = R0 * (t/t0)^v
     * where v is the drift coefficient (typically 0.05-0.1 for amorphous states).
     *
     * For simulation, we model this as additional multiplicative noise
     * that increases with the number of inference cycles.
     */
    void apply_resistance_drift(float* weights, size_t size) {
        // Simulate drift as if time has passed since programming
        // drift_factor represents (t/t0)^v - 1
        static int inference_count = 0;
        inference_count++;

        // Logarithmic time scaling (typical for PCM drift)
        float time_factor = std::log(1.0f + inference_count * 0.01f);
        float drift_scale = config_.pcm_drift_coefficient * time_factor;

        for (size_t i = 0; i < size; ++i) {
            // Drift is more pronounced for intermediate states
            float normalized = std::abs(weights[i]) / 32767.0f;
            float drift_amount = drift_scale * normalized * (1.0f - normalized);
            float epsilon = rng_.normal(0.0f, drift_amount);
            weights[i] *= (1.0f + epsilon);
        }
    }

    /**
     * Read Noise.
     * Optical readout has shot noise and thermal noise.
     * Signal-independent additive Gaussian noise.
     */
    void apply_read_noise(float* result, size_t size) {
        // Find signal magnitude for scaling
        float max_val = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            max_val = std::max(max_val, std::abs(result[i]));
        }
        if (max_val < 1e-10f) return;

        float noise_amplitude = max_val * config_.pcm_read_noise_sigma;
        for (size_t i = 0; i < size; ++i) {
            result[i] += rng_.normal(0.0f, noise_amplitude);
        }
    }
};

} // namespace photonic

#endif // PHOTONIC_CPP_PCM_NOISE_MODEL_H
