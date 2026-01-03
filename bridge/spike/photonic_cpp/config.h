#ifndef PHOTONIC_CPP_CONFIG_H
#define PHOTONIC_CPP_CONFIG_H

#include <cstdlib>
#include <string>
#include <cmath>

namespace photonic {

/**
 * Configuration for photonic models.
 * All parameters can be set via environment variables.
 */
struct PhotonicConfig {
    // Model selection
    std::string model_type = "ideal";  // ideal, mzi_realistic, quantized, all_effects

    // MZI Noise Model Parameters
    float phase_error_sigma = 0.02f;        // FIONA_PHASE_ERROR_SIGMA (default 2%)
    float thermal_crosstalk_sigma = 0.01f;  // FIONA_THERMAL_CROSSTALK_SIGMA (default 1%)
    float insertion_loss_db = 0.3f;         // FIONA_INSERTION_LOSS_DB (dB per stage)
    int num_mzi_stages = 6;                 // FIONA_NUM_MZI_STAGES
    float crosstalk_db = -25.0f;            // FIONA_CROSSTALK_DB
    float detector_noise_sigma = 0.005f;    // FIONA_DETECTOR_NOISE_SIGMA
    int quant_bits = 8;                     // FIONA_QUANT_BITS

    // PCM Noise Model Parameters
    float pcm_prog_variability = 0.03f;     // FIONA_PCM_PROG_VARIABILITY (default 3%)
    float pcm_drift_coefficient = 0.06f;    // FIONA_PCM_DRIFT_COEFF (typical 0.05-0.1)
    float pcm_read_noise_sigma = 0.01f;     // FIONA_PCM_READ_NOISE (default 1%)

    // Debug
    bool debug = false;                     // FIONA_CPP_DEBUG

    // Random seed (0 = use random seed)
    unsigned int random_seed = 0;           // FIONA_RANDOM_SEED

    /**
     * Load configuration from environment variables.
     */
    static PhotonicConfig from_env() {
        PhotonicConfig config;

        // Model type
        const char* model = std::getenv("FIONA_PHOTONIC_MODEL");
        if (model) config.model_type = model;

        // MZI parameters
        const char* val;
        if ((val = std::getenv("FIONA_PHASE_ERROR_SIGMA")))
            config.phase_error_sigma = std::stof(val);
        if ((val = std::getenv("FIONA_THERMAL_CROSSTALK_SIGMA")))
            config.thermal_crosstalk_sigma = std::stof(val);
        if ((val = std::getenv("FIONA_INSERTION_LOSS_DB")))
            config.insertion_loss_db = std::stof(val);
        if ((val = std::getenv("FIONA_NUM_MZI_STAGES")))
            config.num_mzi_stages = std::stoi(val);
        if ((val = std::getenv("FIONA_CROSSTALK_DB")))
            config.crosstalk_db = std::stof(val);
        if ((val = std::getenv("FIONA_DETECTOR_NOISE_SIGMA")))
            config.detector_noise_sigma = std::stof(val);
        if ((val = std::getenv("FIONA_QUANT_BITS")))
            config.quant_bits = std::stoi(val);

        // PCM parameters
        if ((val = std::getenv("FIONA_PCM_PROG_VARIABILITY")))
            config.pcm_prog_variability = std::stof(val);
        if ((val = std::getenv("FIONA_PCM_DRIFT_COEFF")))
            config.pcm_drift_coefficient = std::stof(val);
        if ((val = std::getenv("FIONA_PCM_READ_NOISE")))
            config.pcm_read_noise_sigma = std::stof(val);

        // Debug
        if ((val = std::getenv("FIONA_CPP_DEBUG")))
            config.debug = (std::string(val) == "1");

        // Random seed
        if ((val = std::getenv("FIONA_RANDOM_SEED")))
            config.random_seed = std::stoul(val);

        return config;
    }

    /**
     * Print configuration (for debugging).
     */
    void print() const {
        std::cout << "[PhotonicConfig]" << std::endl;
        std::cout << "  model_type: " << model_type << std::endl;
        std::cout << "  phase_error_sigma: " << phase_error_sigma << std::endl;
        std::cout << "  thermal_crosstalk_sigma: " << thermal_crosstalk_sigma << std::endl;
        std::cout << "  insertion_loss_db: " << insertion_loss_db << std::endl;
        std::cout << "  num_mzi_stages: " << num_mzi_stages << std::endl;
        std::cout << "  crosstalk_db: " << crosstalk_db << std::endl;
        std::cout << "  detector_noise_sigma: " << detector_noise_sigma << std::endl;
        std::cout << "  quant_bits: " << quant_bits << std::endl;
        std::cout << "  pcm_prog_variability: " << pcm_prog_variability << std::endl;
        std::cout << "  pcm_drift_coefficient: " << pcm_drift_coefficient << std::endl;
        std::cout << "  pcm_read_noise_sigma: " << pcm_read_noise_sigma << std::endl;
    }
};

} // namespace photonic

#endif // PHOTONIC_CPP_CONFIG_H
