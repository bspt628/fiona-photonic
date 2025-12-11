import sys
import math
import numpy as np
from numbers import Number


def print_sysinfo():
    print('-' * 20, ' SYSTEM INFORMATION', '-' * 20)
    print(sys.version, '\n')


class Parser:
    def __init__(self, shape_out, matrix_in1, matrix_in2=[],
            bit_out=None, bit_in1=None, bit_in2=None,
            dtype_out=int, dtype_in1=int, dtype_in2=int):
        self.shape_out = shape_out
        self.bit_out = bit_out
        self.bit_in1 = bit_in1
        self.bit_in2 = bit_in2
        self.matrix_in1 = np.array(matrix_in1, dtype=dtype_in1 if self.bit_in1 is None else np.uint8)
        self.matrix_in2 = np.array(matrix_in2, dtype=dtype_in2 if self.bit_in2 is None else np.uint8)
        # 0 - verilator, 1 - spike
        self.mode = 1 if self.bit_out is None else 0

    def _parse_svform(self, arr, bit_length, sign_flag):
        shape_in = arr.shape
        round_flag = True

        # Only use view() for byte-aligned standard sizes (8, 16, 32, 64 bits)
        # Non-standard sizes (e.g., 24-bit) use int.from_bytes
        if bit_length == 8:
            dfmt = "<b" if sign_flag else "<B"
        elif bit_length == 16:
            dfmt = "<h" if sign_flag else "<H"
        elif bit_length == 32:
            dfmt = "<i4" if sign_flag else "<u4"
        elif bit_length == 64:
            dfmt = "<i8" if sign_flag else "<u8"
        else:
            # Non-standard bit lengths (e.g., 24-bit): use int.from_bytes
            round_flag = False
            itemsize = int(math.ceil(bit_length / 8))
            tmp = arr.reshape(-1, itemsize)
            ret = np.array([int.from_bytes(bytes(tmp[i]), byteorder='little', signed=sign_flag) for i in range(0, len(tmp))]).reshape(*shape_in[:-1])

        if round_flag:
            ret = arr.view(dfmt)

        return ret.squeeze()


    def get_in(self, sign_flag1=False, sign_flag2=False):
        if self.mode == 0:
            # verilator
            val1 = self._parse_svform(self.matrix_in1, bit_length=self.bit_in1, sign_flag=sign_flag1)
            if (self.bit_in2 is not None) and (self.matrix_in2 is not None):
                val2 = self._parse_svform(self.matrix_in2, bit_length=self.bit_in2, sign_flag=sign_flag2)
                return val1, val2
            else:
                return val1
            
        else:
            # spike
            if self.matrix_in2.size > 0:
                return self.matrix_in1, self.matrix_in2
            else:
                return self.matrix_in1

    def set_out(self, ret_matrix):
        if isinstance(ret_matrix, Number):
            ret_matrix = np.array([[ret_matrix]])
        if self.mode == 0:
            # verilator
            itemsize = int(math.ceil(self.bit_out / 8))
            ret = ret_matrix.reshape(-1, 1).view(np.uint8)[:, 0:itemsize].reshape(*self.shape_out, itemsize)
            return ret.tolist()

        else:
            # spike
            if np.array(ret_matrix).shape != self.shape_out:
                print('\033[31m' + '[WARN] shape of ret_matrix in Parser.set_out() is NOT expected, which should equal to Parser.shape_out')
            return np.array(ret_matrix).tolist()


class FloatParser:
    """
    Parser for floating-point data types (float32, float16, bfloat16).

    Used for Transformer and other models requiring floating-point precision.
    """

    DTYPE_MAP = {
        'float32': (np.float32, 4),
        'float16': (np.float16, 2),
        'bfloat16': (np.uint16, 2),  # Special handling required
    }

    def __init__(self, shape_out, matrix_in1, matrix_in2=None, dtype='float32'):
        """
        Initialize FloatParser.

        Args:
            shape_out: Output shape specification
            matrix_in1: First input matrix (raw bytes from SV)
            matrix_in2: Second input matrix (raw bytes from SV), optional
            dtype: Data type ('float32', 'float16', 'bfloat16')
        """
        if dtype not in self.DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported: {list(self.DTYPE_MAP.keys())}")

        self.dtype = dtype
        self.np_dtype, self.byte_width = self.DTYPE_MAP[dtype]
        self.shape_out = shape_out

        self.matrix_in1 = self._parse_input(matrix_in1)
        self.matrix_in2 = self._parse_input(matrix_in2) if matrix_in2 is not None else None

    def _parse_input(self, raw_data):
        """
        Parse raw byte data to floating-point array.

        Args:
            raw_data: List of bytes from SystemVerilog via DPI-C

        Returns:
            numpy array of floating-point values
        """
        if raw_data is None:
            return None

        arr = np.array(raw_data, dtype=np.uint8)

        if self.dtype == 'bfloat16':
            return self._bfloat16_to_float32(arr)
        else:
            # float32 or float16: direct view conversion
            return arr.view(self.np_dtype).squeeze()

    def _bfloat16_to_float32(self, arr):
        """
        Convert bfloat16 (stored as uint16) to float32.

        bfloat16 is the upper 16 bits of float32 (sign + exponent + upper 7 mantissa bits).
        """
        bf16 = arr.view(np.uint16)
        fp32_bits = bf16.astype(np.uint32) << 16
        return fp32_bits.view(np.float32).squeeze()

    def _float32_to_bfloat16(self, arr):
        """
        Convert float32 to bfloat16 (stored as uint16).

        Truncates lower 16 bits of mantissa.
        """
        fp32_bits = arr.astype(np.float32).view(np.uint32)
        bf16 = (fp32_bits >> 16).astype(np.uint16)
        return bf16

    def get_in(self):
        """
        Get parsed input arrays.

        Returns:
            Tuple of (matrix_in1, matrix_in2) or just matrix_in1 if matrix_in2 is None
        """
        if self.matrix_in2 is not None:
            return self.matrix_in1, self.matrix_in2
        return self.matrix_in1

    def set_out(self, result):
        """
        Convert result to byte array for return to SystemVerilog.

        Args:
            result: numpy array of computation results

        Returns:
            List of bytes suitable for DPI-C transfer
        """
        if isinstance(result, Number):
            result = np.array([result])

        result = np.asarray(result)

        if self.dtype == 'bfloat16':
            # Convert float32 result to bfloat16 bytes
            bf16 = self._float32_to_bfloat16(result)
            return bf16.view(np.uint8).reshape(*self.shape_out, self.byte_width).tolist()
        else:
            # float32 or float16: direct conversion
            return result.astype(self.np_dtype).view(np.uint8)\
                   .reshape(*self.shape_out, self.byte_width).tolist()
