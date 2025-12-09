import numpy as np
from .utils.parser import Parser
from .utils.formatting import format_matrix


def ops_single(shape_out, matrix_in, bit_out=None, bit_in=None):
    print('[Python] ========== ops_single ==========')
    print(f'[Python] Shape_out = {shape_out}')
    print(format_matrix('Matrix_in (raw)', matrix_in))

    parser = Parser(shape_out, matrix_in, bit_out=bit_out, bit_in1=bit_in)
    comp_in = parser.get_in()

    print(format_matrix('comp_in (parsed)', comp_in))

    comp_out = comp_in.T + 1

    print('[Python] `ops_single` executed: comp_out = comp_in.T + 1')
    print(format_matrix('comp_out (result)', comp_out))
    retval = parser.set_out(comp_out)

    return retval


def ops_dual(shape_out, matrix_in1, matrix_in2, bit_out=None, bit_in1=None, bit_in2=None):
    print('[Python] ========== ops_dual ==========')
    print(f'[Python] Shape_out = {shape_out}')
    print(format_matrix('Matrix_in1 (raw)', matrix_in1))
    print(format_matrix('Matrix_in2 (raw)', matrix_in2))

    parser = Parser(shape_out, matrix_in1, matrix_in2, bit_out=bit_out, bit_in1=bit_in1, bit_in2=bit_in2)
    comp_in1, comp_in2 = parser.get_in()

    print(format_matrix('comp_in1 (parsed)', comp_in1))
    print(format_matrix('comp_in2 (parsed)', comp_in2))

    comp_out = comp_in1.T + comp_in2

    print('[Python] `ops_dual` executed: comp_out = comp_in1.T + comp_in2')
    print(format_matrix('comp_out (result)', comp_out))
    retval = parser.set_out(comp_out)

    return retval