#ifndef FIONA_REGISTER
#define FIONA_REGISTER

#include <utility>
#include <vector>
#include <string>

typedef std::pair<std::string, std::string> PyFileFunc;
typedef std::vector<PyFileFunc> PyFileFuncVec;

const PyFileFuncVec pyfilefunc_reg {
    {"test", "ops_single"},
    {"test", "ops_dual"},
    // Selectable photonic models (controlled by FIONA_PHOTONIC_MODEL env var)
    // Available models: ideal, noisy, quantized, mzi_nonlinear, all_effects
    {"photonic_models", "dotp"},
    {"photonic_models", "mvm"},
    // Floating-point handlers for Transformer models
    {"photonic_models", "mvm_fp32"},           // For Verilator (uses FloatParser with byte data)
    {"photonic_models", "mvm_fp32_spike"},     // For Spike (uses direct float values)
    // Legacy ideal numerical models (kept for reference)
    {"ideal_numerical", "dotp"},
    {"ideal_numerical", "mvm"}
};

#endif /* FIONA_REGISTER */
