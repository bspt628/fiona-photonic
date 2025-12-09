import numpy as np
from .utils.profiler import *
from .utils.parser import Parser
import sys
from .utils.formatting import format_matrix


@profiler()
def dotp(shape_out, matrix_in1, matrix_in2, bit_out=None, bit_in1=None, bit_in2=None):

    print('[Python] Shape_out = ', str(shape_out))
    print('[Python] Matrix_in1 = ', str(matrix_in1))
    print('[Python] Matrix_in2 = ', str(matrix_in2))
    
    parser = Parser(shape_out, matrix_in1, matrix_in2, bit_out=bit_out, bit_in1=bit_in1, bit_in2=bit_in2)
    comp_in1, comp_in2 = parser.get_in()
    comp_in1 = comp_in1.squeeze()
    comp_in2 = comp_in2.squeeze()
    comp_out = np.dot(comp_in1, comp_in2)
    print('[Python] `dotp` executed.')
    print('[Python] comp_out.shape = ', comp_out.shape)
    retval = parser.set_out(comp_out)
    
    return retval


@profiler()
def mvm(shape_out, matrix_in1, matrix_in2, bit_out=None, bit_in1=None, bit_in2=None):

    print('[Python] Shape_out = ', str(shape_out))
    print(format_matrix('Vector_in1 (raw)', matrix_in1))
    print(format_matrix('Matrix_in2 (raw)', matrix_in2))

    parser = Parser(shape_out, matrix_in1, matrix_in2, bit_out=bit_out, bit_in1=bit_in1, bit_in2=bit_in2)
    comp_in1, comp_in2 = parser.get_in()
    comp_out = np.matmul(comp_in2, comp_in1)
    print('[Python] `mvm` executed.')
    print(format_matrix('comp_out (result)', comp_out))
    retval = parser.set_out(comp_out)
    
    return retval
