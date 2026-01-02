#ifndef PHOTONIC_CPP_MODEL_FACTORY_H
#define PHOTONIC_CPP_MODEL_FACTORY_H

#include "photonic_model.h"
#include "ideal_model.h"
#include "mzi_noise_model.h"
#include "config.h"
#include <memory>
#include <iostream>

namespace photonic {

/**
 * Factory for creating photonic models based on configuration.
 */
class ModelFactory {
public:
    /**
     * Create a photonic model based on environment variables.
     *
     * Supported models:
     * - "ideal" (default): Pure mathematical computation
     * - "mzi_realistic": Full MZI noise model
     * - "quantized": DAC/ADC quantization only
     * - "all_effects": Same as mzi_realistic
     *
     * @return Unique pointer to the created model
     */
    static std::unique_ptr<PhotonicModel> create() {
        PhotonicConfig config = PhotonicConfig::from_env();
        return create(config);
    }

    /**
     * Create a photonic model with explicit configuration.
     */
    static std::unique_ptr<PhotonicModel> create(const PhotonicConfig& config) {
        const std::string& model_type = config.model_type;

        if (config.debug) {
            std::cout << "[ModelFactory] Creating model: " << model_type << std::endl;
        }

        if (model_type == "ideal" || model_type.empty()) {
            return std::make_unique<IdealModel>();
        }
        else if (model_type == "mzi_realistic" || model_type == "all_effects") {
            return std::make_unique<MZINoiseModel>(config);
        }
        else if (model_type == "quantized") {
            return std::make_unique<QuantizedModel>(config);
        }
        else {
            std::cerr << "[ModelFactory] Warning: Unknown model type '"
                      << model_type << "', using ideal model" << std::endl;
            return std::make_unique<IdealModel>();
        }
    }

    /**
     * Get singleton instance of the photonic model.
     * Creates the model on first call based on environment variables.
     */
    static PhotonicModel& instance() {
        static std::unique_ptr<PhotonicModel> model = create();
        return *model;
    }
};

} // namespace photonic

#endif // PHOTONIC_CPP_MODEL_FACTORY_H
