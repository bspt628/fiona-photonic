#ifndef PHOTONIC_CPP_NOISE_COMPONENTS_H
#define PHOTONIC_CPP_NOISE_COMPONENTS_H

#include <random>
#include <cmath>
#include <vector>
#include <algorithm>

namespace photonic {
namespace noise {

/**
 * Random number generator wrapper for reproducible noise.
 */
class RandomGenerator {
public:
    RandomGenerator(unsigned int seed = 0) {
        if (seed == 0) {
            std::random_device rd;
            rng_.seed(rd());
        } else {
            rng_.seed(seed);
        }
    }

    float normal(float mean = 0.0f, float stddev = 1.0f) {
        std::normal_distribution<float> dist(mean, stddev);
        return dist(rng_);
    }

    float uniform(float min = 0.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(rng_);
    }

private:
    std::mt19937 rng_;
};

/**
 * Phase Error: Multiplicative Gaussian noise on weights.
 *
 * Models manufacturing variations in MZI phase shifters.
 * W_noisy = W * (1 + epsilon), where epsilon ~ N(0, sigma^2)
 *
 * @param weights Weight matrix (modified in place)
 * @param size Number of elements
 * @param sigma Standard deviation of phase error
 * @param rng Random generator
 */
inline void apply_phase_error(float* weights, size_t size, float sigma, RandomGenerator& rng) {
    for (size_t i = 0; i < size; ++i) {
        float epsilon = rng.normal(0.0f, sigma);
        weights[i] *= (1.0f + epsilon);
    }
}

/**
 * Thermal Crosstalk: Spatially correlated noise via Gaussian filtering.
 *
 * Models heat diffusion between adjacent MZI phase shifters.
 * Applies a simple 2D Gaussian blur to simulate spatial correlation.
 *
 * @param weights Weight matrix (modified in place)
 * @param rows Number of rows
 * @param cols Number of columns
 * @param sigma Crosstalk intensity
 * @param rng Random generator
 */
inline void apply_thermal_crosstalk(float* weights, size_t rows, size_t cols,
                                     float sigma, RandomGenerator& rng) {
    if (sigma <= 0.0f) return;

    // Create noise matrix
    std::vector<float> noise(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        noise[i] = rng.normal(0.0f, sigma);
    }

    // Simple 3x3 Gaussian kernel for spatial correlation
    // kernel = [1, 2, 1; 2, 4, 2; 1, 2, 1] / 16
    std::vector<float> filtered(rows * cols, 0.0f);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int ni = static_cast<int>(i) + di;
                    int nj = static_cast<int>(j) + dj;

                    if (ni >= 0 && ni < static_cast<int>(rows) &&
                        nj >= 0 && nj < static_cast<int>(cols)) {
                        float kernel_weight = (di == 0 && dj == 0) ? 4.0f :
                                              (di == 0 || dj == 0) ? 2.0f : 1.0f;
                        sum += noise[ni * cols + nj] * kernel_weight;
                        weight_sum += kernel_weight;
                    }
                }
            }

            filtered[i * cols + j] = sum / weight_sum;
        }
    }

    // Apply filtered noise to weights
    for (size_t i = 0; i < rows * cols; ++i) {
        weights[i] += filtered[i];
    }
}

/**
 * Insertion Loss: Cumulative optical loss through cascaded stages.
 *
 * Models photon loss at each MZI stage.
 * amplitude_factor = 10^(-total_loss_dB / 20)
 *
 * @param result Output vector (modified in place)
 * @param size Number of elements
 * @param loss_db Loss per stage in dB
 * @param num_stages Number of MZI stages
 */
inline void apply_insertion_loss(float* result, size_t size,
                                  float loss_db, int num_stages) {
    float total_loss_db = loss_db * num_stages;
    float amplitude_factor = std::pow(10.0f, -total_loss_db / 20.0f);

    for (size_t i = 0; i < size; ++i) {
        result[i] *= amplitude_factor;
    }
}

/**
 * Output Crosstalk: Undesired optical coupling between output waveguides.
 *
 * Models parasitic coupling at the output side.
 * Applies a crosstalk matrix: C[i,j] = crosstalk_ratio for i != j
 *
 * @param result Output vector (modified in place)
 * @param size Number of elements
 * @param crosstalk_db Crosstalk level in dB (negative value)
 */
inline void apply_output_crosstalk(float* result, size_t size, float crosstalk_db) {
    if (crosstalk_db >= 0.0f) return;  // No crosstalk

    float crosstalk_ratio = std::pow(10.0f, crosstalk_db / 20.0f);

    // Create copy of original result
    std::vector<float> original(result, result + size);

    // Apply crosstalk
    for (size_t i = 0; i < size; ++i) {
        float crosstalk_sum = 0.0f;
        for (size_t j = 0; j < size; ++j) {
            if (i != j) {
                crosstalk_sum += original[j] * crosstalk_ratio;
            }
        }
        result[i] = original[i] + crosstalk_sum;
    }
}

/**
 * Detector Noise: Shot noise and thermal noise at photodetectors.
 *
 * Models noise at the optical-to-electrical conversion stage.
 * Adds signal-dependent Gaussian noise.
 *
 * @param result Output vector (modified in place)
 * @param size Number of elements
 * @param sigma Noise standard deviation (relative to signal magnitude)
 * @param rng Random generator
 */
inline void apply_detector_noise(float* result, size_t size, float sigma, RandomGenerator& rng) {
    for (size_t i = 0; i < size; ++i) {
        float signal_mag = std::abs(result[i]);
        float noise_std = sigma * (1.0f + signal_mag);  // Signal-dependent noise
        result[i] += rng.normal(0.0f, noise_std);
    }
}

/**
 * DAC/ADC Quantization: Finite resolution of converters.
 *
 * Simulates the limited bit-depth of digital-to-analog and
 * analog-to-digital converters.
 *
 * @param data Data array (modified in place)
 * @param size Number of elements
 * @param bits Number of quantization bits
 * @param is_input True for DAC (input), false for ADC (output)
 */
inline void quantize(float* data, size_t size, int bits, bool is_input) {
    if (bits <= 0 || bits >= 32) return;  // No quantization

    int levels = 1 << bits;  // 2^bits
    float half_levels = levels / 2.0f;

    // Find max absolute value for scaling
    float max_val = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        max_val = std::max(max_val, std::abs(data[i]));
    }
    if (max_val < 1e-10f) return;  // Avoid division by zero

    // Quantize
    for (size_t i = 0; i < size; ++i) {
        // Normalize to [-1, 1]
        float normalized = data[i] / max_val;

        // Scale to quantization levels
        float scaled = normalized * half_levels;

        // Round to nearest integer
        float quantized = std::round(scaled);

        // Clamp to valid range
        quantized = std::max(-half_levels, std::min(half_levels - 1, quantized));

        // Scale back
        data[i] = (quantized / half_levels) * max_val;
    }
}

} // namespace noise
} // namespace photonic

#endif // PHOTONIC_CPP_NOISE_COMPONENTS_H
