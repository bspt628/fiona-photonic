"""
Photonic Models for FIONA Neural Network Accelerator

This module provides various photonic optical element models for MVM operations.
The model type is selected via the model_type parameter passed from SystemVerilog.

Available models:
- ideal: Perfect mathematical operations (default)
- noisy: Gaussian noise added to simulate imperfect components
- quantized: DAC/ADC quantization effects
- mzi_nonlinear: MZI (Mach-Zehnder Interferometer) with non-ideal characteristics
- all_effects: Combined noise, quantization, and nonlinearity

Environment variables (for effect parameters):
- FIONA_NOISE_SIGMA: Noise standard deviation (default: 0.1)
- FIONA_QUANT_BITS: DAC/ADC quantization bits (default: 8)
- FIONA_MZI_EXTINCTION: MZI extinction ratio in dB (default: 30)

Author: FIONA Project
Date: 2025-12-05
"""

import numpy as np
import os
from .utils.profiler import profiler
from .utils.parser import Parser
from .utils.formatting import format_matrix


# ============================================================
# Configuration from environment variables
# ============================================================

def get_model_type():
    """Get the selected photonic model type."""
    return os.environ.get('FIONA_PHOTONIC_MODEL', 'noisy')

def get_noise_sigma():
    """Get the noise standard deviation."""
    return float(os.environ.get('FIONA_NOISE_SIGMA', '0.1'))

def get_quant_bits():
    """Get the DAC/ADC quantization bits."""
    return int(os.environ.get('FIONA_QUANT_BITS', '8'))

def get_mzi_extinction_db():
    """Get the MZI extinction ratio in dB."""
    return float(os.environ.get('FIONA_MZI_EXTINCTION', '30'))


# ============================================================
# Photonic Effect Models
# ============================================================

def apply_noise(value, sigma):
    """Add Gaussian noise to simulate thermal/shot noise in optical components."""
    print(f'[DEBUG apply_noise] value type={type(value)}, sigma={sigma}')
    is_scalar = np.isscalar(value) or (isinstance(value, np.ndarray) and value.shape == ())
    is_integer = np.issubdtype(np.asarray(value).dtype, np.integer)
    value = np.atleast_1d(np.asarray(value))
    print(f'[DEBUG apply_noise] value.shape={value.shape}, is_scalar={is_scalar}')
    noise = np.random.normal(0, sigma, value.shape)
    result = value + noise * np.abs(value).mean()
    # Return scalar if input was scalar
    if is_integer:
        result = np.round(result).astype(np.int64)
    if is_scalar:
        return int(result.flat[0]) if is_integer else float(result.flat[0])
    return result

def apply_quantization(value, bits):
    """Simulate DAC/ADC quantization effects."""
    is_scalar = np.isscalar(value) or (hasattr(value, 'shape') and value.shape == ())
    value = np.atleast_1d(value)
    max_val = np.max(np.abs(value))
    if max_val == 0:
        return value.item() if is_scalar else value

    # Normalize to [-1, 1]
    normalized = value / max_val

    # Quantize
    levels = 2 ** bits
    quantized = np.round(normalized * (levels / 2)) / (levels / 2)

    result = quantized * max_val
    return result.item() if is_scalar else result

def apply_mzi_nonlinearity(value, extinction_db):
    """
    Simulate MZI (Mach-Zehnder Interferometer) non-ideal characteristics.

    MZI transfer function: T = sin^2(phi/2)
    With limited extinction ratio, we model:
    - Non-zero minimum transmission (extinction ratio)
    - Phase-dependent insertion loss
    """
    is_scalar = np.isscalar(value) or (hasattr(value, 'shape') and value.shape == ())
    value = np.atleast_1d(value)

    # Extinction ratio: ratio of max to min transmission
    extinction_ratio = 10 ** (extinction_db / 10)
    t_min = 1 / extinction_ratio

    # Normalize input to phase (assuming value represents intended phase modulation)
    max_val = np.max(np.abs(value))
    if max_val == 0:
        return value.item() if is_scalar else value

    # Add slight nonlinearity based on extinction ratio
    # Higher values get slightly attenuated
    nonlinear_factor = 1.0 - (1.0 - t_min) * (np.abs(value) / max_val) ** 2 * 0.1

    result = value * nonlinear_factor
    return result.item() if is_scalar else result

def apply_all_effects(value, sigma, bits, extinction_db):
    """Apply all photonic effects in sequence."""
    value = apply_quantization(value, bits)
    value = apply_mzi_nonlinearity(value, extinction_db)
    value = apply_noise(value, sigma)
    return value


# ============================================================
# Model Selection and Application
# ============================================================

def apply_photonic_model(result, model_type=None):
    """Apply the selected photonic model to the computation result."""
    if model_type is None:
        model_type = get_model_type()

    print(f'[Photonic Model] Selected model: {model_type}')

    if model_type == 'ideal':
        return result

    elif model_type == 'noisy':
        sigma = get_noise_sigma()
        print(f'[Photonic Model] Applying noise (sigma={sigma}) - TEMPORARILY DISABLED')
        # TODO: Fix apply_noise to work with scalar values from dot product
        return apply_noise(result, sigma)

    elif model_type == 'quantized':
        bits = get_quant_bits()
        print(f'[Photonic Model] Applying quantization ({bits}-bit) - TEMPORARILY DISABLED')
        # TODO: Fix apply_quantization to work with scalar values from dot product
        return result

    elif model_type == 'mzi_nonlinear':
        extinction_db = get_mzi_extinction_db()
        print(f'[Photonic Model] Applying MZI nonlinearity (extinction={extinction_db}dB)')
        return apply_mzi_nonlinearity(result, extinction_db)

    elif model_type == 'all_effects':
        sigma = get_noise_sigma()
        bits = get_quant_bits()
        extinction_db = get_mzi_extinction_db()
        print(f'[Photonic Model] Applying all effects (sigma={sigma}, bits={bits}, extinction={extinction_db}dB)')
        return apply_all_effects(result, sigma, bits, extinction_db)

    else:
        print(f'[Photonic Model] Unknown model "{model_type}", using ideal')
        return result


# ============================================================
# Photonic Operations
# ============================================================

@profiler()
def dotp(shape_out, matrix_in1, matrix_in2, bit_out=None, bit_in1=None, bit_in2=None):
    """Dot product with selectable photonic model."""
    model_type = get_model_type()
    print(f'[Python] ========== dotp (Model: {model_type}) ==========')
    print(f'[Python] Shape_out = {shape_out}')
    print(format_matrix('Matrix_in1 (raw)', matrix_in1))
    print(format_matrix('Matrix_in2 (raw)', matrix_in2))

    parser = Parser(shape_out, matrix_in1, matrix_in2, bit_out=bit_out, bit_in1=bit_in1, bit_in2=bit_in2)
    comp_in1, comp_in2 = parser.get_in()
    comp_in1 = comp_in1.squeeze()
    comp_in2 = comp_in2.squeeze()

    print(format_matrix('comp_in1 (parsed)', comp_in1))
    print(format_matrix('comp_in2 (parsed)', comp_in2))

    # Ideal computation
    comp_out = np.dot(comp_in1, comp_in2)

    # Apply photonic model effects
    comp_out = apply_photonic_model(comp_out, model_type)

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
        model_type: Photonic model type ('ideal', 'noisy', 'quantized', 'mzi_nonlinear', 'all_effects')
                    If None, defaults to 'ideal'
    """
    if model_type is None:
        model_type = 'ideal'
    print(f'[Python] ========== mvm (Model: {model_type}) ==========')
    print(f'[Python] Shape_out = {shape_out}')
    print(format_matrix('Vector_in1 (raw)', matrix_in1))
    print(format_matrix('Matrix_in2 (raw)', matrix_in2))

    parser = Parser(shape_out, matrix_in1, matrix_in2, bit_out=bit_out, bit_in1=bit_in1, bit_in2=bit_in2)
    comp_in1, comp_in2 = parser.get_in()

    print(format_matrix('comp_in1 (parsed)', comp_in1))
    print(format_matrix('comp_in2 (parsed)', comp_in2))

    # Ideal computation
    comp_out = np.matmul(comp_in2, comp_in1)

    # Apply photonic model effects
    comp_out = apply_photonic_model(comp_out, model_type)

    print('[Python] `mvm` executed.')
    print(format_matrix('comp_out (result)', comp_out))
    retval = parser.set_out(comp_out)

    return retval


# ============================================================
# Model Information
# ============================================================

def print_model_info():
    """Print current model configuration."""
    print("=" * 50)
    print("FIONA Photonic Model Configuration")
    print("=" * 50)
    print(f"  Model Type: {get_model_type()}")
    print(f"  Noise Sigma: {get_noise_sigma()}")
    print(f"  Quantization Bits: {get_quant_bits()}")
    print(f"  MZI Extinction: {get_mzi_extinction_db()} dB")
    print("=" * 50)


if __name__ == "__main__":
    print_model_info()
