import operator

import numpy as np

from .. import Quantity, UncertainQuantity
from .. import units as pq
from .common import TestCase, unittest


class TestMatmul(TestCase):

    def test_matrix_multiplication(self):
        q = Quantity([[1, 2, 3], [4, 5, 6]], 'um')
        expected = [[14, 32], [32, 77]] * pq.um**2
        self.assertQuantityEqual(q @ q.T, expected)
        self.assertQuantityEqual(np.matmul(q, q.T), expected)
        self.assertQuantityEqual(q, [[1, 2, 3], [4, 5, 6]] * pq.um)

    def test_vectors_and_broadcasting(self):
        for left_shape, right_shape in (
                ((3,), (3,)),
                ((2, 3), (3,)),
                ((3,), (3, 4)),
                ((2, 3), (3, 4)),
                ((2, 1, 3, 4), (5, 4, 2))):
            with self.subTest(left_shape=left_shape, right_shape=right_shape):
                left = np.arange(np.prod(left_shape)).reshape(left_shape)
                right = np.arange(np.prod(right_shape)).reshape(right_shape)
                expected = np.matmul(left, right) * pq.m * pq.s
                self.assertQuantityEqual((left * pq.m) @ (right * pq.s), expected)
                self.assertQuantityEqual(np.matmul(left * pq.m, right * pq.s), expected)

    def test_plain_array_operands(self):
        left = np.array([[1, 2], [3, 4]])
        right = np.array([[5, 6], [7, 8]])
        expected = [[19, 22], [43, 50]] * pq.m
        for matmul in (operator.matmul, np.matmul):
            for plain_left in (True, False):
                with self.subTest(matmul=matmul, plain_left=plain_left):
                    if plain_left:
                        result = matmul(left, right * pq.m)
                    else:
                        result = matmul(left * pq.m, right)
                    self.assertQuantityEqual(result, expected)

    def test_complex_vectors(self):
        left = Quantity([1 + 2j, 3 - 4j], pq.m)
        right = Quantity([5 - 6j, 7 + 8j], pq.s)
        # matmul, unlike vdot, does not conjugate the first operand.
        self.assertQuantityEqual(left @ right, (70 + 0j) * pq.m * pq.s)

    def test_output_units(self):
        left = [[1., 2.], [3., 4.]] * pq.m
        right = [[5., 6.], [7., 8.]] * pq.s
        out = np.zeros((2, 2)) * pq.kg
        self.assertIs(np.matmul(left, right, out=out), out)
        self.assertQuantityEqual(out, [[19., 22.], [43., 50.]] * pq.m * pq.s)
        self.assertIs(np.matmul(left.magnitude, right.magnitude, out=out), out)
        self.assertQuantityEqual(out, [[19., 22.], [43., 50.]] * pq.dimensionless)

    def test_output_view_units(self):
        left = [[1., 2.], [3., 4.]] * pq.m
        right = [[5., 6.], [7., 8.]] * pq.s
        destination = np.zeros((3, 2)) * pq.m * pq.s
        out = destination[:2]
        self.assertIs(np.matmul(left, right, out=out), out)
        self.assertQuantityEqual(
            destination, [[19., 22.], [43., 50.], [0., 0.]] * pq.m * pq.s)

        invalid = np.zeros((3, 2)) * pq.kg
        with self.assertRaisesRegex(ValueError, 'modify units of a view'):
            np.matmul(left, right, out=invalid[:2])
        self.assertEqual(invalid.units, pq.kg)

    @unittest.skipIf(np.lib.NumpyVersion(np.__version__) < '1.25.0',
                     'NumPy added in-place matrix multiplication in 1.25')
    def test_inplace(self):
        left = [[1., 2.], [3., 4.]] * pq.m
        right = [[5., 6.], [7., 8.]] * pq.s
        left @= right
        self.assertQuantityEqual(left, [[19., 22.], [43., 50.]] * pq.m * pq.s)

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            ([1, 2] * pq.m) @ ([1, 2, 3] * pq.s)
        with self.assertRaises(ValueError):
            (1 * pq.m) @ ([1, 2] * pq.s)

    def test_uncertainty_remains_unsupported(self):
        left = UncertainQuantity([1., 2.], pq.m, uncertainty=[.1, .2])
        right = [3., 4.] * pq.s
        for matmul in (operator.matmul, np.matmul):
            for args in ((left, right), (right, left)):
                with self.subTest(matmul=matmul, args=args):
                    with self.assertRaises(ValueError):
                        matmul(*args)

        for out in (Quantity(0., pq.kg), UncertainQuantity(0., pq.kg, uncertainty=1.)):
            for args in ((left, right), (right, left)):
                with self.subTest(args=args, out_type=type(out)):
                    with self.assertRaises(ValueError):
                        np.matmul(*args, out=out)

        out = UncertainQuantity(0., pq.kg, uncertainty=1.)
        with self.assertRaises(ValueError):
            np.matmul(right, right, out=out)
        with self.assertRaises(ValueError):
            np.matmul(right.magnitude, right.magnitude, out=out)
