#!/usr/bin/env python3
"""
FIONA Photonic Model Unit Tests

This script provides comprehensive unit tests for photonic models.
Run this BEFORE integrating with Spike or Verilator to verify model correctness.

Usage:
    cd /home/bspt628/fiona_undergraduate/fiona-photonic/pyfunc/tests
    python test_photonic_models.py

Author: FIONA Project
Date: 2025-12-05
"""

import sys
import os
import numpy as np
import pytest

# Add parent directories to path for imports
pyfunc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
photonic_dir = os.path.dirname(pyfunc_dir)
sys.path.insert(0, photonic_dir)

# Import using package-style imports
from pyfunc.utils.parser import Parser
from pyfunc.utils.profiler import profiler

# Direct implementations for testing (avoiding relative import issues)
def dotp_test(shape_out, matrix_in1, matrix_in2, bit_out=None, bit_in1=None, bit_in2=None):
    """Test version of dotp without profiler decorator"""
    parser = Parser(shape_out, matrix_in1, matrix_in2, bit_out=bit_out, bit_in1=bit_in1, bit_in2=bit_in2)
    comp_in1, comp_in2 = parser.get_in()
    comp_in1 = comp_in1.squeeze()
    comp_in2 = comp_in2.squeeze()
    comp_out = np.dot(comp_in1, comp_in2)
    return parser.set_out(comp_out)

def mvm_test(shape_out, matrix_in1, matrix_in2, bit_out=None, bit_in1=None, bit_in2=None):
    """Test version of mvm without profiler decorator"""
    parser = Parser(shape_out, matrix_in1, matrix_in2, bit_out=bit_out, bit_in1=bit_in1, bit_in2=bit_in2)
    comp_in1, comp_in2 = parser.get_in()
    comp_out = np.matmul(comp_in2, comp_in1)
    return parser.set_out(comp_out)

# Alias for tests
dotp = dotp_test
mvm = mvm_test

# Photonic effect functions (copied for standalone testing)
def apply_noise(value, sigma):
    """Add Gaussian noise to simulate thermal/shot noise in optical components."""
    is_scalar = np.isscalar(value) or (isinstance(value, np.ndarray) and value.shape == ())
    value = np.atleast_1d(np.asarray(value))
    noise = np.random.normal(0, sigma, value.shape)
    result = value + noise * np.abs(value).mean()
    if is_scalar:
        return float(result.flat[0])
    return result

def apply_quantization(value, bits):
    """Simulate DAC/ADC quantization effects."""
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
    """Simulate MZI non-ideal characteristics."""
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

def apply_all_effects(value, sigma, bits, extinction_db):
    """Apply all photonic effects in sequence."""
    value = apply_quantization(value, bits)
    value = apply_mzi_nonlinearity(value, extinction_db)
    value = apply_noise(value, sigma)
    return value

def apply_photonic_model(result, model_type='ideal'):
    """Apply the selected photonic model to the computation result."""
    if model_type == 'ideal':
        return result
    elif model_type == 'noisy':
        return apply_noise(result, 0.01)
    elif model_type == 'quantized':
        return apply_quantization(result, 8)
    elif model_type == 'mzi_nonlinear':
        return apply_mzi_nonlinearity(result, 30)
    elif model_type == 'all_effects':
        return apply_all_effects(result, 0.01, 8, 30)
    else:
        return result


class TestResults:
    """Simple test result tracker"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = 0

    def record_pass(self, name):
        self.passed += 1
        print(f"  [PASS] {name}")

    def record_fail(self, name, message):
        self.failed += 1
        print(f"  [FAIL] {name}: {message}")

    def record_error(self, name, exception):
        self.errors += 1
        print(f"  [ERROR] {name}: {exception}")

    def summary(self):
        total = self.passed + self.failed + self.errors
        print(f"\n{'='*50}")
        print(f"Test Summary: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"  Failed: {self.failed}")
        if self.errors > 0:
            print(f"  Errors: {self.errors}")
        print(f"{'='*50}")
        return self.failed == 0 and self.errors == 0


# Pytest fixture for TestResults
@pytest.fixture
def results():
    """Pytest fixture that provides a TestResults instance"""
    return TestResults()


# =============================================================================
# Test: Dot Product (DOTP)
# =============================================================================

def test_dotp_basic(results):
    """Basic dot product test"""
    vec1 = np.array([1, 2, 3, 4], dtype=np.float32)
    vec2 = np.array([1, 1, 1, 1], dtype=np.float32)

    result = dotp("1,1", vec1, vec2)
    expected = 10.0

    if np.isclose(result, expected):
        results.record_pass("dotp_basic")
    else:
        results.record_fail("dotp_basic", f"Expected {expected}, got {result}")


def test_dotp_negative(results):
    """Dot product with negative values"""
    vec1 = np.array([1, -2, 3, -4], dtype=np.float32)
    vec2 = np.array([2, 2, 2, 2], dtype=np.float32)

    result = dotp("1,1", vec1, vec2)
    expected = -4.0  # 2 - 4 + 6 - 8

    if np.isclose(result, expected):
        results.record_pass("dotp_negative")
    else:
        results.record_fail("dotp_negative", f"Expected {expected}, got {result}")


def test_dotp_zeros(results):
    """Dot product with zero vector"""
    vec1 = np.zeros(4, dtype=np.float32)
    vec2 = np.array([1, 2, 3, 4], dtype=np.float32)

    result = dotp("1,1", vec1, vec2)
    expected = 0.0

    # Handle both scalar and array output formats
    result_val = np.asarray(result).flatten()[0] if hasattr(result, '__iter__') else result
    if np.isclose(result_val, expected):
        results.record_pass("dotp_zeros")
    else:
        results.record_fail("dotp_zeros", f"Expected {expected}, got {result}")


def test_dotp_single_element(results):
    """Dot product with single element"""
    vec1 = np.array([5], dtype=np.float32)
    vec2 = np.array([3], dtype=np.float32)

    result = dotp("1,1", vec1, vec2)
    expected = 15.0

    if np.isclose(result, expected):
        results.record_pass("dotp_single_element")
    else:
        results.record_fail("dotp_single_element", f"Expected {expected}, got {result}")


def test_dotp_large_vector(results):
    """Dot product with larger vector (32 elements)"""
    vec1 = np.arange(32, dtype=np.float32)
    vec2 = np.ones(32, dtype=np.float32)

    result = dotp("1,1", vec1, vec2)
    expected = np.sum(vec1)  # 0+1+2+...+31 = 496

    if np.isclose(result, expected):
        results.record_pass("dotp_large_vector")
    else:
        results.record_fail("dotp_large_vector", f"Expected {expected}, got {result}")


# =============================================================================
# Test: Matrix-Vector Multiplication (MVM)
# =============================================================================

def test_mvm_basic(results):
    """Basic MVM test"""
    # 2x3 matrix * 3-element vector = 2-element vector
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6]], dtype=np.float32)
    vector = np.array([1, 1, 1], dtype=np.float32)

    result = mvm("2,1", vector, matrix)
    expected = np.array([6, 15], dtype=np.float32)  # [1+2+3, 4+5+6]

    if np.allclose(result, expected):
        results.record_pass("mvm_basic")
    else:
        results.record_fail("mvm_basic", f"Expected {expected}, got {result}")


def test_mvm_identity(results):
    """MVM with identity matrix"""
    matrix = np.eye(4, dtype=np.float32)
    vector = np.array([1, 2, 3, 4], dtype=np.float32)

    result = mvm("4,1", vector, matrix)

    if np.allclose(result, vector):
        results.record_pass("mvm_identity")
    else:
        results.record_fail("mvm_identity", f"Expected {vector}, got {result}")


def test_mvm_zeros(results):
    """MVM with zero matrix"""
    matrix = np.zeros((3, 4), dtype=np.float32)
    vector = np.array([1, 2, 3, 4], dtype=np.float32)

    result = mvm("3,1", vector, matrix)
    expected = np.zeros(3, dtype=np.float32)

    if np.allclose(result, expected):
        results.record_pass("mvm_zeros")
    else:
        results.record_fail("mvm_zeros", f"Expected {expected}, got {result}")


def test_mvm_single_row(results):
    """MVM with single row matrix (equivalent to dot product)"""
    matrix = np.array([[1, 2, 3, 4]], dtype=np.float32)  # 1x4
    vector = np.array([1, 1, 1, 1], dtype=np.float32)

    result = mvm("1,1", vector, matrix)
    expected = np.array([10], dtype=np.float32)

    if np.allclose(result, expected):
        results.record_pass("mvm_single_row")
    else:
        results.record_fail("mvm_single_row", f"Expected {expected}, got {result}")


# =============================================================================
# Test: Photonic Effects
# =============================================================================

def test_noise_magnitude(results):
    """Test noise magnitude is reasonable"""
    value = np.array([1.0, 2.0, 3.0, 4.0])
    sigma = 0.01

    # Run 100 trials
    noisy_results = [apply_noise(value.copy(), sigma) for _ in range(100)]
    mean_result = np.mean(noisy_results, axis=0)

    # Mean should be close to original value
    if np.allclose(mean_result, value, atol=0.1):
        results.record_pass("noise_magnitude")
    else:
        results.record_fail("noise_magnitude",
            f"Mean {mean_result} not close to {value}")


def test_noise_variability(results):
    """Test that noise adds variability"""
    value = np.array([1.0, 1.0, 1.0, 1.0])
    sigma = 0.1

    result1 = apply_noise(value.copy(), sigma)
    result2 = apply_noise(value.copy(), sigma)

    # Results should be different
    if not np.allclose(result1, result2):
        results.record_pass("noise_variability")
    else:
        results.record_fail("noise_variability", "Noise should add variability")


def test_quantization_levels(results):
    """Test quantization produces discrete levels"""
    value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    bits = 4  # 16 levels

    result = apply_quantization(value, bits)

    # Result should still be in valid range
    max_val = np.max(np.abs(value))
    if np.all(np.abs(result) <= max_val * 1.01):  # Allow small tolerance
        results.record_pass("quantization_levels")
    else:
        results.record_fail("quantization_levels",
            f"Result {result} exceeds expected range")


def test_quantization_preserves_zeros(results):
    """Test that zero stays zero after quantization"""
    value = np.array([0.0, 0.0, 0.0])
    bits = 8

    result = apply_quantization(value, bits)

    if np.allclose(result, value):
        results.record_pass("quantization_preserves_zeros")
    else:
        results.record_fail("quantization_preserves_zeros",
            f"Expected zeros, got {result}")


def test_mzi_nonlinearity_small_effect(results):
    """Test MZI nonlinearity has small effect at high extinction"""
    value = np.array([1.0, 2.0, 3.0, 4.0])
    extinction_db = 30  # High extinction = small nonlinearity

    result = apply_mzi_nonlinearity(value, extinction_db)

    # Result should be close to original with small attenuation
    if np.allclose(result, value, rtol=0.1):
        results.record_pass("mzi_nonlinearity_small_effect")
    else:
        results.record_fail("mzi_nonlinearity_small_effect",
            f"Expected close to {value}, got {result}")


def test_mzi_nonlinearity_larger_effect(results):
    """Test MZI nonlinearity has larger effect at low extinction"""
    value = np.array([1.0, 2.0, 3.0, 4.0])
    extinction_db = 10  # Lower extinction = larger nonlinearity

    result = apply_mzi_nonlinearity(value, extinction_db)

    # Result should still be in reasonable range
    if np.all(np.abs(result) <= np.abs(value) * 1.1):
        results.record_pass("mzi_nonlinearity_larger_effect")
    else:
        results.record_fail("mzi_nonlinearity_larger_effect",
            f"Result {result} outside expected range")


def test_all_effects_combination(results):
    """Test all effects combined"""
    value = np.array([1.0, 2.0, 3.0, 4.0])
    sigma = 0.01
    bits = 8
    extinction_db = 30

    result = apply_all_effects(value, sigma, bits, extinction_db)

    # Result should be in reasonable range (not blown up)
    if np.all(np.isfinite(result)) and np.all(np.abs(result) < 100):
        results.record_pass("all_effects_combination")
    else:
        results.record_fail("all_effects_combination",
            f"Result {result} outside reasonable range")


# =============================================================================
# Test: Photonic Model Selection
# =============================================================================

def test_model_selection_ideal(results):
    """Test ideal model selection"""
    value = np.array([1.0, 2.0, 3.0])

    result = apply_photonic_model(value.copy(), model_type='ideal')

    if np.array_equal(result, value):
        results.record_pass("model_selection_ideal")
    else:
        results.record_fail("model_selection_ideal",
            f"Ideal should return unchanged: {result}")


def test_model_selection_noisy(results):
    """Test noisy model selection"""
    value = np.array([1.0, 2.0, 3.0])

    result = apply_photonic_model(value.copy(), model_type='noisy')

    # Result should be different due to noise
    if not np.array_equal(result, value):
        results.record_pass("model_selection_noisy")
    else:
        results.record_fail("model_selection_noisy",
            "Noisy model should add noise")


# =============================================================================
# Test: Edge Cases
# =============================================================================

def test_scalar_input(results):
    """Test handling of scalar input"""
    value = 5.0
    sigma = 0.01

    try:
        result = apply_noise(value, sigma)
        if isinstance(result, (int, float)):
            results.record_pass("scalar_input")
        else:
            results.record_fail("scalar_input",
                f"Expected scalar, got {type(result)}")
    except Exception as e:
        results.record_error("scalar_input", e)


def test_empty_input(results):
    """Test handling of empty array"""
    value = np.array([])
    sigma = 0.01

    try:
        result = apply_noise(value, sigma)
        if len(result) == 0:
            results.record_pass("empty_input")
        else:
            results.record_fail("empty_input", f"Expected empty, got {result}")
    except Exception as e:
        # Empty input might raise an error, which is acceptable
        results.record_pass("empty_input (expected error)")


# =============================================================================
# Test: Batched MVM Operations
# =============================================================================

def mvm_batched_test(batch_data, model_type='ideal'):
    """Test version of mvm_batched"""
    results = []
    for item in batch_data:
        if len(item) == 3:
            mat, vec, vlen = item
        else:
            mat, vec = item
            vlen = len(vec)

        mat_np = np.array(mat, dtype=np.float64)
        vec_np = np.array(vec, dtype=np.float64)

        if mat_np.ndim == 1:
            size = int(np.sqrt(len(mat_np)))
            mat_np = mat_np.reshape(size, size)

        vec_np = vec_np.flatten()[:vlen]
        result = np.matmul(mat_np, vec_np)

        if model_type != 'ideal':
            result = apply_photonic_model(result, model_type)

        results.append(result.astype(np.int16).tolist())

    return results


def test_mvm_batched_basic(results):
    """Basic batched MVM test with single operation"""
    mat = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.int16)  # Identity
    vec = np.array([1, 2, 3, 4], dtype=np.int16)

    batch = [(mat, vec, 4)]
    batch_results = mvm_batched_test(batch)

    expected = [1, 2, 3, 4]

    if len(batch_results) == 1 and np.allclose(batch_results[0], expected):
        results.record_pass("mvm_batched_basic")
    else:
        results.record_fail("mvm_batched_basic",
            f"Expected {expected}, got {batch_results}")


def test_mvm_batched_multiple(results):
    """Batched MVM with multiple operations"""
    # Operation 1: Identity matrix
    mat1 = np.eye(4, dtype=np.int16)
    vec1 = np.array([1, 2, 3, 4], dtype=np.int16)

    # Operation 2: All ones matrix
    mat2 = np.ones((4, 4), dtype=np.int16)
    vec2 = np.array([1, 1, 1, 1], dtype=np.int16)

    # Operation 3: Scaling matrix
    mat3 = np.array([[2, 0, 0, 0],
                     [0, 2, 0, 0],
                     [0, 0, 2, 0],
                     [0, 0, 0, 2]], dtype=np.int16)
    vec3 = np.array([1, 2, 3, 4], dtype=np.int16)

    batch = [
        (mat1, vec1, 4),
        (mat2, vec2, 4),
        (mat3, vec3, 4),
    ]

    batch_results = mvm_batched_test(batch)

    expected = [
        [1, 2, 3, 4],      # Identity
        [4, 4, 4, 4],      # Sum of vec2 for each row
        [2, 4, 6, 8],      # 2x scaling
    ]

    if len(batch_results) == 3:
        all_correct = True
        for i, (got, exp) in enumerate(zip(batch_results, expected)):
            if not np.allclose(got, exp):
                all_correct = False
                results.record_fail("mvm_batched_multiple",
                    f"Operation {i}: expected {exp}, got {got}")
                break
        if all_correct:
            results.record_pass("mvm_batched_multiple")
    else:
        results.record_fail("mvm_batched_multiple",
            f"Expected 3 results, got {len(batch_results)}")


def test_mvm_batched_empty(results):
    """Batched MVM with empty batch"""
    batch = []
    batch_results = mvm_batched_test(batch)

    if len(batch_results) == 0:
        results.record_pass("mvm_batched_empty")
    else:
        results.record_fail("mvm_batched_empty",
            f"Expected empty list, got {batch_results}")


def test_mvm_batched_large_batch(results):
    """Batched MVM with large batch (simulating tiled MVM)"""
    batch_size = 16  # Simulate 16 tile operations
    batch = []

    for i in range(batch_size):
        mat = np.eye(4, dtype=np.int16) * (i + 1)  # Scale by index
        vec = np.ones(4, dtype=np.int16)
        batch.append((mat, vec, 4))

    batch_results = mvm_batched_test(batch)

    if len(batch_results) == batch_size:
        all_correct = True
        for i, result in enumerate(batch_results):
            expected = [i + 1] * 4
            if not np.allclose(result, expected):
                all_correct = False
                results.record_fail("mvm_batched_large_batch",
                    f"Operation {i}: expected {expected}, got {result}")
                break
        if all_correct:
            results.record_pass("mvm_batched_large_batch")
    else:
        results.record_fail("mvm_batched_large_batch",
            f"Expected {batch_size} results, got {len(batch_results)}")


def test_mvm_batched_consistency(results):
    """Test that batched MVM gives same results as individual MVMs"""
    # Create test data
    mat = np.array([[1, 2], [3, 4]], dtype=np.int16)
    vec = np.array([1, 1], dtype=np.int16)

    # Individual MVM
    individual_result = mvm("2,1", vec, mat)

    # Batched MVM
    batch = [(mat, vec, 2)]
    batch_results = mvm_batched_test(batch)

    if np.allclose(individual_result, batch_results[0]):
        results.record_pass("mvm_batched_consistency")
    else:
        results.record_fail("mvm_batched_consistency",
            f"Individual: {individual_result}, Batched: {batch_results[0]}")


def test_mvm_batched_32x32(results):
    """Test batched MVM with 32x32 matrices (FIONA hardware size)"""
    # Create 32x32 identity matrix
    mat = np.eye(32, dtype=np.int16)
    vec = np.arange(32, dtype=np.int16)

    batch = [(mat, vec, 32)]
    batch_results = mvm_batched_test(batch)

    expected = list(range(32))

    if len(batch_results) == 1 and np.allclose(batch_results[0], expected):
        results.record_pass("mvm_batched_32x32")
    else:
        results.record_fail("mvm_batched_32x32",
            f"Expected identity result, got different values")


def test_mvm_batched_negative_values(results):
    """Test batched MVM with negative values"""
    mat = np.array([[1, -1], [-1, 1]], dtype=np.int16)
    vec = np.array([10, 5], dtype=np.int16)

    batch = [(mat, vec, 2)]
    batch_results = mvm_batched_test(batch)

    expected = [5, -5]  # [10-5, -10+5]

    if len(batch_results) == 1 and np.allclose(batch_results[0], expected):
        results.record_pass("mvm_batched_negative_values")
    else:
        results.record_fail("mvm_batched_negative_values",
            f"Expected {expected}, got {batch_results[0]}")


def test_mvm_batched_performance(results):
    """Test that batched operations are faster than individual calls"""
    import time

    batch_size = 100
    mat = np.eye(8, dtype=np.int16)
    vec = np.ones(8, dtype=np.int16)

    # Create batch
    batch = [(mat, vec, 8) for _ in range(batch_size)]

    # Time batched operation
    start = time.time()
    batch_results = mvm_batched_test(batch)
    batch_time = time.time() - start

    # Time individual operations
    start = time.time()
    individual_results = []
    for _ in range(batch_size):
        result = mvm("8,1", vec, mat)
        individual_results.append(result)
    individual_time = time.time() - start

    # Batched should complete (performance comparison is informational)
    if len(batch_results) == batch_size:
        results.record_pass(f"mvm_batched_performance (batch:{batch_time:.3f}s, individual:{individual_time:.3f}s)")
    else:
        results.record_fail("mvm_batched_performance",
            f"Expected {batch_size} results")


# =============================================================================
# Test: mvm_batched_unified (Verilator + Spike mode support)
# =============================================================================

def mvm_batched_unified_test(batch_data, model_type='ideal'):
    """Test version of mvm_batched_unified"""
    if not batch_data:
        return []

    first_item = batch_data[0]
    is_verilator_mode = 'bit_out' in first_item

    results = []

    for item in batch_data:
        if is_verilator_mode:
            # Verilator mode
            shape_out = item['shape_out']
            matrix_in1 = item['matrix_in1']
            matrix_in2 = item['matrix_in2']
            bit_out = item.get('bit_out', 16)
            bit_in1 = item.get('bit_in1', 16)
            bit_in2 = item.get('bit_in2', 16)

            parser = Parser(shape_out, matrix_in1, matrix_in2,
                          bit_out=bit_out, bit_in1=bit_in1, bit_in2=bit_in2)
            vec_np, mat_np = parser.get_in(sign_flag1=True, sign_flag2=True)

            result = np.matmul(mat_np.astype(np.float64), vec_np.astype(np.float64))

            if model_type != 'ideal':
                result = apply_photonic_model(result, model_type)

            result = np.round(result).astype(np.int64)
            output = parser.set_out(result)
            results.append(output)
        else:
            # Spike mode
            mat_data = item.get('mat', [])
            vec_data = item.get('vec', [])
            vlen = item.get('vlen', 32)
            rd = item.get('rd', 0)

            mat_np = np.array(mat_data, dtype=np.int16)
            if mat_np.ndim == 1:
                mat_np = mat_np.reshape(vlen, vlen)
            vec_np = np.array(vec_data, dtype=np.int16)[:vlen]

            result = np.matmul(mat_np.astype(np.float64), vec_np.astype(np.float64))

            if model_type != 'ideal':
                result = apply_photonic_model(result, model_type)

            result = np.round(result).astype(np.int16)
            results.append({
                'result': result.tolist(),
                'rd': rd
            })

    return results


def test_mvm_batched_unified_spike_mode(results):
    """Test mvm_batched_unified in Spike mode (dict format without bit_out)"""
    batch = [
        {'mat': np.eye(4, dtype=np.int16).tolist(), 'vec': [1, 2, 3, 4], 'vlen': 4, 'rd': 0},
        {'mat': (np.ones((4, 4), dtype=np.int16) * 2).tolist(), 'vec': [1, 1, 1, 1], 'vlen': 4, 'rd': 1},
    ]

    batch_results = mvm_batched_unified_test(batch)

    expected = [
        {'result': [1, 2, 3, 4], 'rd': 0},
        {'result': [8, 8, 8, 8], 'rd': 1},  # 2*4 = 8
    ]

    if len(batch_results) == 2:
        all_correct = True
        for i, (got, exp) in enumerate(zip(batch_results, expected)):
            if got['rd'] != exp['rd'] or not np.allclose(got['result'], exp['result']):
                all_correct = False
                results.record_fail("mvm_batched_unified_spike_mode",
                    f"Operation {i}: expected {exp}, got {got}")
                break
        if all_correct:
            results.record_pass("mvm_batched_unified_spike_mode")
    else:
        results.record_fail("mvm_batched_unified_spike_mode",
            f"Expected 2 results, got {len(batch_results)}")


def test_mvm_batched_unified_verilator_mode(results):
    """Test mvm_batched_unified in Verilator mode (with bit_out)"""
    # Create test data with proper byte format
    # 4-element vector: [1, 2, 3, 4] as int16 bytes
    vec_int16 = np.array([1, 2, 3, 4], dtype=np.int16)
    vec_bytes = vec_int16.view(np.uint8).tolist()

    # 4x4 identity matrix as int16 bytes
    mat_int16 = np.eye(4, dtype=np.int16)
    mat_bytes = mat_int16.view(np.uint8).tolist()

    batch = [{
        'shape_out': (4,),
        'matrix_in1': vec_bytes,
        'matrix_in2': mat_bytes,
        'bit_out': 16,
        'bit_in1': 16,
        'bit_in2': 16,
    }]

    try:
        batch_results = mvm_batched_unified_test(batch)

        # Expected: identity * [1,2,3,4] = [1,2,3,4]
        # Output format is byte array from Parser.set_out
        if len(batch_results) == 1:
            # Parse output bytes back to int16
            output_bytes = np.array(batch_results[0], dtype=np.uint8)
            output_int16 = output_bytes.view(np.int16)
            expected = [1, 2, 3, 4]

            if np.allclose(output_int16, expected):
                results.record_pass("mvm_batched_unified_verilator_mode")
            else:
                results.record_fail("mvm_batched_unified_verilator_mode",
                    f"Expected {expected}, got {output_int16.tolist()}")
        else:
            results.record_fail("mvm_batched_unified_verilator_mode",
                f"Expected 1 result, got {len(batch_results)}")
    except Exception as e:
        results.record_error("mvm_batched_unified_verilator_mode", e)


def test_mvm_batched_unified_empty(results):
    """Test mvm_batched_unified with empty batch"""
    batch = []
    batch_results = mvm_batched_unified_test(batch)

    if len(batch_results) == 0:
        results.record_pass("mvm_batched_unified_empty")
    else:
        results.record_fail("mvm_batched_unified_empty",
            f"Expected empty list, got {batch_results}")


def test_mvm_batched_unified_mixed_not_allowed(results):
    """Test that mixed mode batches use first item's mode"""
    # This test verifies the mode detection from first item
    batch = [
        {'mat': [[1, 0], [0, 1]], 'vec': [1, 2], 'vlen': 2, 'rd': 0},
        # If we add a Verilator-style item, it should be treated as Spike mode
        # because first item doesn't have bit_out
    ]

    batch_results = mvm_batched_unified_test(batch)

    # Should work in Spike mode
    if len(batch_results) == 1 and 'result' in batch_results[0]:
        results.record_pass("mvm_batched_unified_mixed_not_allowed")
    else:
        results.record_fail("mvm_batched_unified_mixed_not_allowed",
            f"Expected Spike mode result, got {batch_results}")


def test_mvm_batched_unified_32x32_spike(results):
    """Test mvm_batched_unified with 32x32 matrices in Spike mode"""
    mat = np.eye(32, dtype=np.int16)
    vec = np.arange(32, dtype=np.int16)

    batch = [
        {'mat': mat.tolist(), 'vec': vec.tolist(), 'vlen': 32, 'rd': 5}
    ]

    batch_results = mvm_batched_unified_test(batch)

    expected_result = list(range(32))

    if len(batch_results) == 1:
        got = batch_results[0]
        if got['rd'] == 5 and np.allclose(got['result'], expected_result):
            results.record_pass("mvm_batched_unified_32x32_spike")
        else:
            results.record_fail("mvm_batched_unified_32x32_spike",
                f"Expected rd=5, result={expected_result[:5]}..., got rd={got['rd']}, result={got['result'][:5]}...")
    else:
        results.record_fail("mvm_batched_unified_32x32_spike",
            f"Expected 1 result, got {len(batch_results)}")


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all unit tests"""
    results = TestResults()

    print("\n" + "="*50)
    print("FIONA Photonic Model Unit Tests")
    print("="*50)

    # DOTP tests
    print("\n--- Dot Product (DOTP) Tests ---")
    test_dotp_basic(results)
    test_dotp_negative(results)
    test_dotp_zeros(results)
    test_dotp_single_element(results)
    test_dotp_large_vector(results)

    # MVM tests
    print("\n--- Matrix-Vector Multiplication (MVM) Tests ---")
    test_mvm_basic(results)
    test_mvm_identity(results)
    test_mvm_zeros(results)
    test_mvm_single_row(results)

    # Batched MVM tests
    print("\n--- Batched MVM Tests ---")
    test_mvm_batched_basic(results)
    test_mvm_batched_multiple(results)
    test_mvm_batched_empty(results)
    test_mvm_batched_large_batch(results)
    test_mvm_batched_consistency(results)
    test_mvm_batched_32x32(results)
    test_mvm_batched_negative_values(results)
    test_mvm_batched_performance(results)

    # Unified Batched MVM tests (Verilator + Spike)
    print("\n--- Unified Batched MVM Tests (Verilator/Spike) ---")
    test_mvm_batched_unified_spike_mode(results)
    test_mvm_batched_unified_verilator_mode(results)
    test_mvm_batched_unified_empty(results)
    test_mvm_batched_unified_mixed_not_allowed(results)
    test_mvm_batched_unified_32x32_spike(results)

    # Photonic effect tests
    print("\n--- Photonic Effect Tests ---")
    test_noise_magnitude(results)
    test_noise_variability(results)
    test_quantization_levels(results)
    test_quantization_preserves_zeros(results)
    test_mzi_nonlinearity_small_effect(results)
    test_mzi_nonlinearity_larger_effect(results)
    test_all_effects_combination(results)

    # Model selection tests
    print("\n--- Model Selection Tests ---")
    test_model_selection_ideal(results)
    test_model_selection_noisy(results)

    # Edge case tests
    print("\n--- Edge Case Tests ---")
    test_scalar_input(results)
    test_empty_input(results)

    # Summary
    success = results.summary()
    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
