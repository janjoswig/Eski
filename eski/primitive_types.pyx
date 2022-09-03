import numpy as np


P_AVALUE = np.float64
P_AVALUE32 = np.float32
P_AINDEX = np.intp
P_ABOOL = np.uint8

cpdef Constants make_constants():
    cdef Constants constants = Constants(
        1.380649e-26,         # kB in kJ/K
        0.00831446261815324,  # R in kJ/(K mol)
        1.66053906660e-27,    # u in kg
        6.0221408e+23,        # nA
        299792458,            # c in m/s
        1.602176634e-19,      # ze in C
        8.8541878128e-12,     # e0 in F/m
        138.935458            # (kJ nm)/(mol ze^2)
    )

    return constants

cdef AVALUE* _allocate_and_fill_avalue_array(AINDEX n, list values):

    cdef AVALUE *ptr
    cdef AINDEX i

    ptr = <AVALUE*>malloc(n * sizeof(AVALUE))

    if ptr == NULL:
        raise MemoryError()

    for i in range(n):
        ptr[i] = values[i]

    return ptr

cdef AINDEX* _allocate_and_fill_aindex_array(AINDEX n, list values):

    cdef AINDEX *ptr
    cdef AINDEX i

    ptr = <AINDEX*>malloc(n * sizeof(AINDEX))

    if ptr == NULL:
        raise MemoryError()

    for i in range(n):
        ptr[i] = values[i]

    return ptr