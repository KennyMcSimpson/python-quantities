====================
Quick-start tutorial
====================

.. automodule:: quantities

Matrix multiplication
=====================

The ``@`` operator and ``numpy.matmul`` multiply the units of their operands,
just like elementwise multiplication::

    >>> import quantities as pq
    >>> left = pq.Quantity([[1, 2], [3, 4]], 'm')
    >>> right = pq.Quantity([[5, 6], [7, 8]], 's')
    >>> result = left @ right
    >>> result.magnitude.tolist()
    [[19, 22], [43, 50]]
    >>> bool(result.units == pq.m * pq.s)
    True

Uncertainty propagation for matrix multiplication is not supported.
