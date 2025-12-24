"""
Photonic Models for FIONA Neural Network Accelerator

This module provides various photonic optical element models for MVM operations.
The model type is selected via the model_type parameter passed from SystemVerilog.

Available models:
- ideal: Perfect mathematical operations (default)
- noisy: Simple Gaussian noise (legacy, for backward compatibility)
- mzi_realistic: Realistic MZI-based model with phase error, loss, and crosstalk
- quantized: DAC/ADC quantization effects
- mzi_nonlinear: MZI with non-ideal characteristics (legacy)
- all_effects: Combined all realistic effects

Realistic MZI Model Parameters (environment variables):
- FIONA_PHASE_ERROR_SIGMA: Phase error std dev (default: 0.02 = 2%)
- FIONA_THERMAL_CROSSTALK_SIGMA: Thermal crosstalk std dev (default: 0.01 = 1%)
- FIONA_INSERTION_LOSS_DB: Insertion loss per MZI stage in dB (default: 0.3)
- FIONA_CROSSTALK_DB: Output crosstalk in dB (default: -25)
- FIONA_NUM_MZI_STAGES: Number of MZI stages (default: 6)
- FIONA_DETECTOR_NOISE_SIGMA: Detector noise std dev (default: 0.005)
- FIONA_QUANT_BITS: DAC/ADC quantization bits (default: 8)

References:
- Shen et al., "Deep learning with coherent nanophotonic circuits", Nature Photonics 2017
- arXiv:2308.03249 "Analysis of Optical Loss and Crosstalk Noise in MZI-based Coherent Photonic Neural Networks"
- arXiv:2504.05685 "Fast Prediction of Phase shifters Phase Error Distributions Using Gaussian Processes"
- arXiv:2404.10589 "Thermal Crosstalk Modelling and Compensation Methods"

Author: FIONA Project
Date: 2025-12-10
"""

import numpy as np
import os
import gc
from scipy.ndimage import gaussian_filter
from .utils.profiler import profiler
from .utils.parser import Parser
from .utils.formatting import format_matrix

# GC optimization settings
# Set FIONA_GC_INTERVAL to control garbage collection frequency
# 0 = every call (slowest, lowest memory)
# N = every N calls (balance)
# -1 = disabled (fastest, highest memory)
_gc_call_counter = 0

def get_gc_interval():
    """Get GC interval. Set FIONA_GC_INTERVAL environment variable."""
    return int(os.environ.get('FIONA_GC_INTERVAL', '10'))


# Verbose logging control - only log first N calls to avoid flooding
_verbose_call_counter = 0
_verbose_max_logs = 3  # Only log first 3 calls
_verbose_model_printed = False  # Print model info once

def get_verbose_limit():
    """Get max number of verbose log entries. Set FIONA_VERBOSE_LIMIT env var."""
    return int(os.environ.get('FIONA_VERBOSE_LIMIT', '3'))


# ============================================================
# Configuration from environment variables
# ============================================================

def get_verbose():
    """Get verbose output flag. Set FIONA_VERBOSE=1 to enable debug output."""
    return os.environ.get('FIONA_VERBOSE', '0') == '1'

def should_log_verbose():
    """Check if we should log verbose output (limited to first N calls)."""
    global _verbose_call_counter, _verbose_max_logs
    if not get_verbose():
        return False
    _verbose_max_logs = get_verbose_limit()
    if _verbose_call_counter < _verbose_max_logs:
        _verbose_call_counter += 1
        return True
    return False

def log_model_info_once():
    """Print model info once at the start."""
    global _verbose_model_printed
    if get_verbose() and not _verbose_model_printed:
        model_type = get_model_type()
        print(f'[Python] ========== FIONA Photonic Model: {model_type} ==========')
        if model_type in ['mzi_realistic', 'all_effects']:
            print(f'[Python]   Phase error: {get_phase_error_sigma()*100:.1f}%')
            print(f'[Python]   Insertion loss: {get_insertion_loss_db():.2f} dB/stage')
            print(f'[Python]   Thermal crosstalk: {get_thermal_crosstalk_sigma()*100:.1f}%')
        _verbose_model_printed = True

def get_model_type():
    """Get the selected photonic model type."""
    return os.environ.get('FIONA_PHOTONIC_MODEL', 'ideal')

# Legacy parameters
def get_noise_sigma():
    """Get the noise standard deviation (legacy)."""
    return float(os.environ.get('FIONA_NOISE_SIGMA', '0.1'))

def get_quant_bits():
    """Get the DAC/ADC quantization bits."""
    return int(os.environ.get('FIONA_QUANT_BITS', '8'))

def get_mzi_extinction_db():
    """Get the MZI extinction ratio in dB (legacy)."""
    return float(os.environ.get('FIONA_MZI_EXTINCTION', '30'))

# Realistic MZI model parameters
def get_phase_error_sigma():
    """Get phase error standard deviation.

    Typical values (from arXiv:2504.05685):
    - High quality: 0.005 (0.5%)
    - Standard: 0.02 (2%)
    - Low quality: 0.05 (5%)
    """
    return float(os.environ.get('FIONA_PHASE_ERROR_SIGMA', '0.02'))

def get_thermal_crosstalk_sigma():
    """Get thermal crosstalk standard deviation.

    Typical values (from arXiv:2404.10589):
    - Well isolated: 0.005
    - Standard: 0.01-0.03
    - Dense packing: 0.05
    """
    return float(os.environ.get('FIONA_THERMAL_CROSSTALK_SIGMA', '0.01'))

def get_insertion_loss_db():
    """Get insertion loss per MZI stage in dB.

    Typical values (from IEEE/arXiv:2308.03249):
    - Optimized: 0.2 dB
    - Standard: 0.3-0.5 dB
    - High loss: 1.0 dB
    """
    return float(os.environ.get('FIONA_INSERTION_LOSS_DB', '0.3'))

def get_crosstalk_db():
    """Get output crosstalk in dB.

    Typical values (from arXiv:2308.03249):
    - High quality: -38 dB
    - Standard: -25 dB
    - Low quality: -20 dB
    """
    return float(os.environ.get('FIONA_CROSSTALK_DB', '-25'))

def get_num_mzi_stages():
    """Get number of MZI stages for loss calculation."""
    return int(os.environ.get('FIONA_NUM_MZI_STAGES', '6'))

def get_detector_noise_sigma():
    """Get detector noise standard deviation (shot + thermal noise)."""
    return float(os.environ.get('FIONA_DETECTOR_NOISE_SIGMA', '0.005'))


# ============================================================
# Realistic MZI Noise Model Components
# ============================================================

class MZINoiseModel:
    """
    Realistic MZI-based photonic neural network noise model.

    This model incorporates:
    1. Phase error (fabrication variance) - multiplicative noise on weights
    2. Thermal crosstalk - spatially correlated noise between adjacent MZIs
    3. Insertion loss - cumulative optical loss through MZI stages
    4. Output crosstalk - interference between output channels
    5. Detector noise - shot noise and thermal noise at photodetectors
    6. DAC/ADC quantization - discretization effects

    References:
    - Shen et al., Nature Photonics 2017
    - arXiv:2308.03249, arXiv:2504.05685, arXiv:2404.10589
    """

    def __init__(self,
                 phase_error_sigma=None,
                 thermal_crosstalk_sigma=None,
                 insertion_loss_db=None,
                 crosstalk_db=None,
                 num_stages=None,
                 detector_noise_sigma=None,
                 quant_bits=None):
        """Initialize MZI noise model with parameters."""
        self.phase_error_sigma = phase_error_sigma or get_phase_error_sigma()
        self.thermal_crosstalk_sigma = thermal_crosstalk_sigma or get_thermal_crosstalk_sigma()
        self.insertion_loss_db = insertion_loss_db or get_insertion_loss_db()
        self.crosstalk_db = crosstalk_db or get_crosstalk_db()
        self.num_stages = num_stages or get_num_mzi_stages()
        self.detector_noise_sigma = detector_noise_sigma or get_detector_noise_sigma()
        self.quant_bits = quant_bits or get_quant_bits()

    def apply_phase_error(self, W):
        """
        Apply phase error to weight matrix (multiplicative Gaussian noise).

        Models fabrication-induced phase variations in MZI arms.
        Phase error follows Gaussian distribution: W_noisy = W * (1 + epsilon)
        where epsilon ~ N(0, sigma^2)

        Reference: arXiv:2504.05685 - Phase error follows multivariate Gaussian
        """
        if self.phase_error_sigma == 0:
            return W

        epsilon = np.random.normal(0, self.phase_error_sigma, W.shape)
        W_noisy = W * (1 + epsilon)
        return W_noisy

    def apply_thermal_crosstalk(self, W):
        """
        Apply thermal crosstalk with spatial correlation.

        Models heat diffusion between adjacent MZI phase shifters.
        Uses Gaussian spatial filtering to create correlated noise.

        Reference: arXiv:2404.10589 - Thermal crosstalk modelling
        """
        if self.thermal_crosstalk_sigma == 0:
            return W

        # Generate base noise
        noise = np.random.normal(0, self.thermal_crosstalk_sigma, W.shape)

        # Apply spatial correlation (Gaussian kernel simulates heat diffusion)
        # Correlation length ~1-2 elements for typical MZI spacing
        if W.ndim >= 2:
            correlated_noise = gaussian_filter(noise, sigma=1.0)
        else:
            correlated_noise = noise

        return W + correlated_noise * np.abs(W).mean()

    def apply_insertion_loss(self, y):
        """
        Apply cumulative insertion loss through MZI stages.

        Total loss = loss_per_stage * num_stages
        Power ratio = 10^(-loss_dB/10)
        Amplitude ratio = 10^(-loss_dB/20)

        Reference: arXiv:2308.03249 - Optical loss analysis
        """
        if self.insertion_loss_db == 0 or self.num_stages == 0:
            return y

        total_loss_db = self.insertion_loss_db * self.num_stages
        amplitude_factor = 10 ** (-total_loss_db / 20)

        return y * amplitude_factor

    def apply_output_crosstalk(self, y):
        """
        Apply crosstalk between output channels.

        Models undesired optical coupling between output waveguides.
        Crosstalk matrix: C[i,j] = crosstalk_ratio for i != j, 1 for i == j

        Reference: arXiv:2308.03249 - Crosstalk noise analysis
        """
        if self.crosstalk_db <= -60:  # Negligible crosstalk
            return y

        n = len(y)
        crosstalk_ratio = 10 ** (self.crosstalk_db / 20)

        # Crosstalk matrix: diagonal = 1, off-diagonal = crosstalk_ratio
        C = np.eye(n) + crosstalk_ratio * (np.ones((n, n)) - np.eye(n))

        return np.matmul(C, y)

    def apply_detector_noise(self, y):
        """
        Apply detector noise (shot noise + thermal noise).

        Models photodetector imperfections.
        Additive Gaussian noise scaled by signal magnitude.
        """
        if self.detector_noise_sigma == 0:
            return y

        # Signal-dependent noise (approximates shot noise behavior)
        noise = np.random.normal(0, self.detector_noise_sigma, y.shape)
        noise_scaled = noise * (np.abs(y).mean() + 1)  # +1 to avoid zero scaling

        return y + noise_scaled

    def apply_quantization(self, value, is_input=True):
        """
        Apply DAC (input) or ADC (output) quantization.

        Models finite resolution of digital-to-analog and analog-to-digital converters.
        """
        if self.quant_bits >= 16:  # Effectively no quantization
            return value

        max_val = np.max(np.abs(value))
        if max_val == 0:
            return value

        # Normalize, quantize, denormalize
        levels = 2 ** self.quant_bits
        normalized = value / max_val
        quantized = np.round(normalized * (levels / 2)) / (levels / 2)

        return quantized * max_val

    def forward(self, W, x, apply_dac=True, apply_adc=True, verbose=False):
        """
        Perform MVM with realistic noise model.

        Pipeline:
        1. DAC quantization on inputs (optional)
        2. Phase error on weights
        3. Thermal crosstalk on weights
        4. Matrix-vector multiplication
        5. Insertion loss
        6. Output crosstalk
        7. Detector noise
        8. ADC quantization on output (optional)

        Args:
            W: Weight matrix (2D)
            x: Input vector (1D)
            apply_dac: Whether to apply DAC quantization
            apply_adc: Whether to apply ADC quantization
            verbose: Print debug information

        Returns:
            y: Output vector with noise effects
        """
        if verbose:
            print(f'[MZI Model] Parameters:')
            print(f'  Phase error sigma: {self.phase_error_sigma}')
            print(f'  Thermal crosstalk sigma: {self.thermal_crosstalk_sigma}')
            print(f'  Insertion loss: {self.insertion_loss_db} dB/stage x {self.num_stages} stages')
            print(f'  Output crosstalk: {self.crosstalk_db} dB')
            print(f'  Detector noise sigma: {self.detector_noise_sigma}')
            print(f'  Quantization bits: {self.quant_bits}')

        # 1. DAC quantization on input
        if apply_dac:
            x = self.apply_quantization(x, is_input=True)
            if verbose:
                print(f'[MZI Model] After DAC: x range = [{x.min():.2f}, {x.max():.2f}]')

        # 2. Phase error on weights
        W_noisy = self.apply_phase_error(W)
        if verbose:
            diff = np.abs(W_noisy - W).mean()
            print(f'[MZI Model] After phase error: mean |delta_W| = {diff:.4f}')

        # 3. Thermal crosstalk on weights
        W_noisy = self.apply_thermal_crosstalk(W_noisy)
        if verbose:
            diff = np.abs(W_noisy - W).mean()
            print(f'[MZI Model] After thermal crosstalk: mean |delta_W| = {diff:.4f}')

        # 4. Matrix-vector multiplication (core computation)
        y = np.matmul(W_noisy, x)
        if verbose:
            print(f'[MZI Model] After MVM: y range = [{y.min():.2f}, {y.max():.2f}]')

        # 5. Insertion loss
        y = self.apply_insertion_loss(y)
        if verbose:
            print(f'[MZI Model] After insertion loss: y range = [{y.min():.2f}, {y.max():.2f}]')

        # 6. Output crosstalk
        y = self.apply_output_crosstalk(y)
        if verbose:
            print(f'[MZI Model] After crosstalk: y range = [{y.min():.2f}, {y.max():.2f}]')

        # 7. Detector noise
        y = self.apply_detector_noise(y)
        if verbose:
            print(f'[MZI Model] After detector noise: y range = [{y.min():.2f}, {y.max():.2f}]')

        # 8. ADC quantization on output
        if apply_adc:
            y = self.apply_quantization(y, is_input=False)
            if verbose:
                print(f'[MZI Model] After ADC: y range = [{y.min():.2f}, {y.max():.2f}]')

        return y


# ============================================================
# Legacy Photonic Effect Models (for backward compatibility)
# ============================================================

def apply_noise(value, sigma):
    """Add Gaussian noise to simulate thermal/shot noise (legacy model)."""
    is_scalar = np.isscalar(value) or (isinstance(value, np.ndarray) and value.shape == ())
    is_integer = np.issubdtype(np.asarray(value).dtype, np.integer)
    value = np.atleast_1d(np.asarray(value))
    noise = np.random.normal(0, sigma, value.shape)
    result = value + noise * np.abs(value).mean()
    if is_integer:
        result = np.round(result).astype(np.int64)
    if is_scalar:
        return int(result.flat[0]) if is_integer else float(result.flat[0])
    return result

def apply_quantization(value, bits):
    """Simulate DAC/ADC quantization effects (legacy model)."""
    is_scalar = np.isscalar(value) or (hasattr(value, 'shape') and value.shape == ())
    value = np.atleast_1d(value)
    max_val = np.max(np.abs(value))
    if max_val == 0:
        return value.item() if is_scalar else value
    normalized = value / max_val
    levels = 2 ** bits
    quantized = np.round(normalized * (levels / 2)) / (levels / 2)
    result = quantized * max_val
    return result.item() if is_scalar else result

def apply_mzi_nonlinearity(value, extinction_db):
    """Simulate MZI non-ideal characteristics (legacy model)."""
    is_scalar = np.isscalar(value) or (hasattr(value, 'shape') and value.shape == ())
    value = np.atleast_1d(value)
    extinction_ratio = 10 ** (extinction_db / 10)
    t_min = 1 / extinction_ratio
    max_val = np.max(np.abs(value))
    if max_val == 0:
        return value.item() if is_scalar else value
    nonlinear_factor = 1.0 - (1.0 - t_min) * (np.abs(value) / max_val) ** 2 * 0.1
    result = value * nonlinear_factor
    return result.item() if is_scalar else result


# ============================================================
# Model Selection and Application
# ============================================================

def apply_photonic_model(result, model_type=None, W=None, x=None):
    """
    Apply the selected photonic model.

    Args:
        result: Ideal computation result (used for legacy models)
        model_type: Model type string
        W: Weight matrix (for realistic model)
        x: Input vector (for realistic model)

    Returns:
        Noisy result
    """
    if model_type is None:
        model_type = get_model_type()

    if model_type == 'ideal':
        return result

    elif model_type == 'noisy':
        return apply_noise(result, get_noise_sigma())

    elif model_type == 'mzi_realistic':
        if W is None or x is None:
            # Fallback: apply phase error as simple noise
            return apply_noise(result, get_phase_error_sigma())
        model = MZINoiseModel()
        return model.forward(W, x, apply_dac=True, apply_adc=True)

    elif model_type == 'quantized':
        return apply_quantization(result, get_quant_bits())

    elif model_type == 'mzi_nonlinear':
        return apply_mzi_nonlinearity(result, get_mzi_extinction_db())

    elif model_type == 'all_effects':
        if W is None or x is None:
            value = apply_quantization(result, get_quant_bits())
            return apply_noise(value, get_noise_sigma())
        model = MZINoiseModel()
        return model.forward(W, x, apply_dac=True, apply_adc=True)

    else:
        return result


# ============================================================
# Photonic Operations
# ============================================================

@profiler()
def dotp(shape_out, matrix_in1, matrix_in2, bit_out=None, bit_in1=None, bit_in2=None):
    """Dot product with selectable photonic model."""
    global _gc_call_counter
    model_type = get_model_type()
    log_model_info_once()

    parser = Parser(shape_out, matrix_in1, matrix_in2, bit_out=bit_out, bit_in1=bit_in1, bit_in2=bit_in2)
    comp_in1, comp_in2 = parser.get_in()
    comp_in1 = comp_in1.squeeze()
    comp_in2 = comp_in2.squeeze()

    # For mzi_realistic: treat as 1-row matrix MVM: dot(a,b) = [a] @ b
    if model_type in ['mzi_realistic', 'all_effects']:
        W = comp_in1.reshape(1, -1)  # 1 x N matrix
        x = comp_in2  # N vector
        result = apply_photonic_model(None, model_type, W=W, x=x)
        # Get scalar value
        comp_out = np.asarray(result).flatten()[0]
    else:
        comp_out = np.dot(comp_in1, comp_in2)
        comp_out = apply_photonic_model(comp_out, model_type)

    # INT16演算の場合、整数に変換（mvm関数と同様）
    if np.issubdtype(comp_in1.dtype, np.integer):
        comp_out = int(np.round(comp_out))

    retval = parser.set_out(comp_out)

    # 定期的なGC実行（メモリリーク防止）
    gc_interval = get_gc_interval()
    if gc_interval >= 0:
        _gc_call_counter += 1
        if gc_interval == 0 or _gc_call_counter >= gc_interval:
            del parser, comp_in1, comp_in2
            gc.collect()
            _gc_call_counter = 0

    return retval


@profiler()
def mvm(shape_out, matrix_in1, matrix_in2, bit_out=None, bit_in1=None, bit_in2=None, model_type=None):
    """Matrix-vector multiplication with selectable photonic model."""
    global _gc_call_counter
    if model_type is None:
        model_type = get_model_type()
    log_model_info_once()  # Print model info once at start

    parser = Parser(shape_out, matrix_in1, matrix_in2, bit_out=bit_out, bit_in1=bit_in1, bit_in2=bit_in2)
    # Use signed interpretation for input arrays (2's complement)
    comp_in1, comp_in2 = parser.get_in(sign_flag1=True, sign_flag2=True)

    # For realistic models, pass W and x to apply_photonic_model
    if model_type in ['mzi_realistic', 'all_effects']:
        # Realistic model: noise applied to weights before MVM
        comp_out = apply_photonic_model(None, model_type, W=comp_in2, x=comp_in1)
    else:
        # Legacy models: noise applied to result after MVM
        comp_out_ideal = np.matmul(comp_in2, comp_in1)
        comp_out = apply_photonic_model(comp_out_ideal, model_type)

    # Ensure integer output for compatibility with SV
    if np.issubdtype(comp_in1.dtype, np.integer):
        comp_out = np.round(comp_out).astype(np.int64)

    retval = parser.set_out(comp_out)

    # 定期的なGC実行（メモリリーク防止）
    gc_interval = get_gc_interval()
    if gc_interval >= 0:
        _gc_call_counter += 1
        if gc_interval == 0 or _gc_call_counter >= gc_interval:
            del parser, comp_in1, comp_in2
            gc.collect()
            _gc_call_counter = 0

    return retval


# ============================================================
# Floating-Point Operations (for Transformer models)
# ============================================================

@profiler()
def mvm_fp32(shape_out, matrix_in1, matrix_in2, dtype='float32', model_type='ideal'):
    """Matrix-vector multiplication with 32-bit floating-point precision."""
    from .utils.parser import FloatParser
    log_model_info_once()
    verbose = should_log_verbose()

    if verbose:
        print(f'[Python] mvm_fp32: shape_out={shape_out}, model={model_type}')

    parser = FloatParser(shape_out, matrix_in1, matrix_in2, dtype=dtype)
    vec, mat = parser.get_in()

    # Matrix-vector multiplication with photonic model
    if model_type in ['mzi_realistic', 'all_effects']:
        model = MZINoiseModel()
        result = model.forward(mat, vec, apply_dac=False, apply_adc=False)
    else:
        result = np.matmul(mat, vec)
        if model_type != 'ideal':
            result = apply_photonic_model(result, model_type)

    return parser.set_out(result)


@profiler()
def mvm_fp32_spike(shape_out, matrix_in1, matrix_in2):
    """Matrix-vector multiplication for Spike simulator (float values directly)."""
    global _gc_call_counter
    model_type = get_model_type()
    log_model_info_once()
    verbose = should_log_verbose()

    if verbose:
        print(f'[Python] mvm_fp32_spike: shape_out={shape_out}')

    # Convert input lists to numpy arrays
    vec = np.array(matrix_in1, dtype=np.float32).flatten()
    mat = np.array(matrix_in2, dtype=np.float32)

    # Matrix-vector multiplication with photonic model
    if model_type in ['mzi_realistic', 'all_effects']:
        model = MZINoiseModel()
        result = model.forward(mat, vec, apply_dac=False, apply_adc=False)
    elif model_type == 'noisy':
        result = np.matmul(mat, vec)
        result = result + np.random.normal(0, get_noise_sigma(), result.shape)
    else:
        result = np.matmul(mat, vec)

    # Convert to output format
    output = [[float(v)] for v in result]

    # Periodic garbage collection
    gc_interval = get_gc_interval()
    if gc_interval >= 0:
        _gc_call_counter += 1
        if gc_interval == 0 or _gc_call_counter >= gc_interval:
            del vec, mat, result
            gc.collect()
            _gc_call_counter = 0

    return output


# ============================================================
# Model Information
# ============================================================

def print_model_info():
    """Print current model configuration."""
    print("=" * 60)
    print("FIONA Photonic Model Configuration")
    print("=" * 60)
    print(f"  Model Type: {get_model_type()}")
    print()
    print("  Legacy Parameters:")
    print(f"    Noise Sigma: {get_noise_sigma()}")
    print(f"    MZI Extinction: {get_mzi_extinction_db()} dB")
    print()
    print("  Realistic MZI Parameters:")
    print(f"    Phase Error Sigma: {get_phase_error_sigma()} ({get_phase_error_sigma()*100:.1f}%)")
    print(f"    Thermal Crosstalk Sigma: {get_thermal_crosstalk_sigma()} ({get_thermal_crosstalk_sigma()*100:.1f}%)")
    print(f"    Insertion Loss: {get_insertion_loss_db()} dB/stage")
    print(f"    Output Crosstalk: {get_crosstalk_db()} dB")
    print(f"    Num MZI Stages: {get_num_mzi_stages()}")
    print(f"    Detector Noise Sigma: {get_detector_noise_sigma()} ({get_detector_noise_sigma()*100:.2f}%)")
    print(f"    Quantization Bits: {get_quant_bits()}")
    print("=" * 60)


if __name__ == "__main__":
    print_model_info()
