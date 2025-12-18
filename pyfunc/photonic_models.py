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
from scipy.ndimage import gaussian_filter
from .utils.profiler import profiler
from .utils.parser import Parser
from .utils.formatting import format_matrix


# ============================================================
# Configuration from environment variables
# ============================================================

def get_verbose():
    """Get verbose output flag. Set FIONA_VERBOSE=1 to enable debug output."""
    return os.environ.get('FIONA_VERBOSE', '0') == '1'

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
    verbose = get_verbose()
    if model_type is None:
        model_type = get_model_type()

    if verbose:
        print(f'[Photonic Model] Selected model: {model_type}')

    if model_type == 'ideal':
        return result

    elif model_type == 'noisy':
        # Legacy simple noise model
        sigma = get_noise_sigma()
        if verbose:
            print(f'[Photonic Model] Applying legacy noise (sigma={sigma})')
        return apply_noise(result, sigma)

    elif model_type == 'mzi_realistic':
        # New realistic MZI model
        if W is None or x is None:
            if verbose:
                print('[Photonic Model] WARNING: W and x required for mzi_realistic, falling back to legacy')
            return apply_noise(result, get_phase_error_sigma())

        model = MZINoiseModel()
        if verbose:
            print(f'[Photonic Model] Applying realistic MZI model:')
            print(f'  - Phase error: {model.phase_error_sigma*100:.1f}%')
            print(f'  - Thermal crosstalk: {model.thermal_crosstalk_sigma*100:.1f}%')
            print(f'  - Insertion loss: {model.insertion_loss_db:.2f} dB/stage x {model.num_stages} stages')
            print(f'  - Output crosstalk: {model.crosstalk_db:.1f} dB')
            print(f'  - Detector noise: {model.detector_noise_sigma*100:.2f}%')
        return model.forward(W, x, apply_dac=True, apply_adc=True)

    elif model_type == 'quantized':
        bits = get_quant_bits()
        if verbose:
            print(f'[Photonic Model] Applying quantization ({bits}-bit)')
        return apply_quantization(result, bits)

    elif model_type == 'mzi_nonlinear':
        extinction_db = get_mzi_extinction_db()
        if verbose:
            print(f'[Photonic Model] Applying MZI nonlinearity (extinction={extinction_db}dB)')
        return apply_mzi_nonlinearity(result, extinction_db)

    elif model_type == 'all_effects':
        # Use realistic model with all effects
        if W is None or x is None:
            if verbose:
                print('[Photonic Model] WARNING: W and x required for all_effects, using legacy')
            sigma = get_noise_sigma()
            bits = get_quant_bits()
            value = apply_quantization(result, bits)
            return apply_noise(value, sigma)

        model = MZINoiseModel()
        if verbose:
            print(f'[Photonic Model] Applying all realistic MZI effects')
        return model.forward(W, x, apply_dac=True, apply_adc=True, verbose=verbose)

    else:
        if verbose:
            print(f'[Photonic Model] Unknown model "{model_type}", using ideal')
        return result


# ============================================================
# Photonic Operations
# ============================================================

@profiler()
def dotp(shape_out, matrix_in1, matrix_in2, bit_out=None, bit_in1=None, bit_in2=None):
    """Dot product with selectable photonic model."""
    model_type = get_model_type()
    verbose = get_verbose()

    if verbose:
        print(f'[Python] ========== dotp (Model: {model_type}) ==========')
        print(f'[Python] Shape_out = {shape_out}')
        print(format_matrix('Matrix_in1 (raw)', matrix_in1))
        print(format_matrix('Matrix_in2 (raw)', matrix_in2))

    parser = Parser(shape_out, matrix_in1, matrix_in2, bit_out=bit_out, bit_in1=bit_in1, bit_in2=bit_in2)
    comp_in1, comp_in2 = parser.get_in()
    comp_in1 = comp_in1.squeeze()
    comp_in2 = comp_in2.squeeze()

    if verbose:
        print(format_matrix('comp_in1 (parsed)', comp_in1))
        print(format_matrix('comp_in2 (parsed)', comp_in2))

    # Ideal computation
    comp_out = np.dot(comp_in1, comp_in2)

    # Apply photonic model effects
    comp_out = apply_photonic_model(comp_out, model_type)

    if verbose:
        print('[Python] `dotp` executed.')
        print(format_matrix('comp_out (result)', comp_out))

    retval = parser.set_out(comp_out)
    return retval


@profiler()
def mvm(shape_out, matrix_in1, matrix_in2, bit_out=None, bit_in1=None, bit_in2=None, model_type=None):
    """Matrix-vector multiplication with selectable photonic model.

    Args:
        shape_out: Output shape specification
        matrix_in1: Input vector (1D)
        matrix_in2: Input matrix (2D)
        bit_out: Output bit width
        bit_in1: Input1 bit width
        bit_in2: Input2 bit width
        model_type: Photonic model type:
            - 'ideal': Perfect mathematical operations
            - 'noisy': Legacy simple Gaussian noise
            - 'mzi_realistic': Realistic MZI model with all effects
            - 'quantized': DAC/ADC quantization only
            - 'all_effects': Alias for mzi_realistic with verbose output
    """
    if model_type is None:
        model_type = 'ideal'
    verbose = get_verbose()

    if verbose:
        print(f'[Python] ========== mvm (Model: {model_type}) ==========')
        print(f'[Python] Shape_out = {shape_out}')
        print(format_matrix('Vector_in1 (raw)', matrix_in1))
        print(format_matrix('Matrix_in2 (raw)', matrix_in2))

    parser = Parser(shape_out, matrix_in1, matrix_in2, bit_out=bit_out, bit_in1=bit_in1, bit_in2=bit_in2)
    # Use signed interpretation for input arrays (2's complement)
    comp_in1, comp_in2 = parser.get_in(sign_flag1=True, sign_flag2=True)

    if verbose:
        print(format_matrix('comp_in1 (parsed, vector x)', comp_in1))
        print(format_matrix('comp_in2 (parsed, matrix W)', comp_in2))

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

    if verbose:
        print('[Python] `mvm` executed.')
        print(format_matrix('comp_out (result)', comp_out))

    retval = parser.set_out(comp_out)
    return retval


# ============================================================
# Floating-Point Operations (for Transformer models)
# ============================================================

@profiler()
def mvm_fp32(shape_out, matrix_in1, matrix_in2, dtype='float32', model_type='ideal'):
    """
    Matrix-vector multiplication with 32-bit floating-point precision.

    Designed for Transformer models where floating-point precision is required
    to handle outliers in attention mechanisms.

    Args:
        shape_out: Output shape specification
        matrix_in1: Input vector (1D, float32 bytes)
        matrix_in2: Weight matrix (2D, float32 bytes)
        dtype: Data type (currently only 'float32' supported)
        model_type: Photonic model type ('ideal', 'mzi_realistic', etc.)

    Returns:
        Output vector as list of float32 bytes
    """
    from .utils.parser import FloatParser
    verbose = get_verbose()

    if verbose:
        print(f'[Python] ========== mvm_fp32 (dtype: {dtype}, model: {model_type}) ==========')
        print(f'[Python] Shape_out = {shape_out}')

    # Parse floating-point inputs
    parser = FloatParser(shape_out, matrix_in1, matrix_in2, dtype=dtype)
    vec, mat = parser.get_in()

    if verbose:
        print(f'[Python] Input vector shape: {vec.shape}, dtype: {vec.dtype}')
        print(f'[Python] Input vector: {vec}')
        print(f'[Python] Weight matrix shape: {mat.shape}, dtype: {mat.dtype}')
        print(f'[Python] Weight matrix:\n{mat}')

    # Matrix-vector multiplication with photonic model
    if model_type in ['mzi_realistic', 'all_effects']:
        model = MZINoiseModel()
        if verbose:
            print(f'[Python] Applying MZI realistic model:')
            print(f'  - Phase error: {model.phase_error_sigma*100:.1f}%')
            print(f'  - Insertion loss: {model.insertion_loss_db:.2f} dB/stage x {model.num_stages} stages')
        result = model.forward(mat, vec, apply_dac=False, apply_adc=False)
    else:
        # Ideal computation: y = W @ x
        result = np.matmul(mat, vec)
        if model_type != 'ideal':
            result = apply_photonic_model(result, model_type)

    if verbose:
        print(f'[Python] Output vector shape: {result.shape}, dtype: {result.dtype}')
        print(f'[Python] Output vector: {result}')

    return parser.set_out(result)


@profiler()
def mvm_fp32_spike(shape_out, matrix_in1, matrix_in2):
    """
    Matrix-vector multiplication with 32-bit floating-point precision.

    This version is designed for Spike simulator which passes float values directly
    (not as raw bytes). Uses the same interface as the int16 mvm function.

    Args:
        shape_out: Output shape specification (rows, cols)
        matrix_in1: Input vector (1D, float values)
        matrix_in2: Weight matrix (2D, float values)

    Returns:
        Output vector as list of lists of float values
    """
    model_type = get_model_type()
    verbose = get_verbose()

    if verbose:
        print(f'[Python] ========== mvm_fp32_spike (model: {model_type}) ==========')
        print(f'[Python] Shape_out = {shape_out}')

    # Convert input lists to numpy arrays
    vec = np.array(matrix_in1, dtype=np.float32).flatten()
    mat = np.array(matrix_in2, dtype=np.float32)

    if verbose:
        print(f'[Python] Input vector shape: {vec.shape}')
        print(f'[Python] Input vector (first 8): {vec[:8]}...')
        print(f'[Python] Weight matrix shape: {mat.shape}')

    # Matrix-vector multiplication with photonic model
    if model_type in ['mzi_realistic', 'all_effects']:
        model = MZINoiseModel()
        if verbose:
            print(f'[Python] Applying MZI realistic model')
        result = model.forward(mat, vec, apply_dac=False, apply_adc=False)
    elif model_type == 'noisy':
        result = np.matmul(mat, vec)
        result = result + np.random.normal(0, get_noise_sigma(), result.shape)
    else:
        # Ideal computation: y = W @ x
        result = np.matmul(mat, vec)

    if verbose:
        print(f'[Python] Output vector shape: {result.shape}')
        print(f'[Python] Output vector (first 8): {result[:8]}...')

    # Return as nested list [[v0], [v1], ...] to match expected format
    return [[float(v)] for v in result]


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
