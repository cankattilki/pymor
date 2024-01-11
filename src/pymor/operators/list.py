# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.base import abstractmethod
from pymor.operators.interface import Operator


class ListVectorArrayOperatorBase(Operator):
    """Base |Operator| for |ListVectorArrays|."""

    def _prepare_apply(self, U, mu, kind, least_squares=False):
        pass

    @abstractmethod
    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        pass

    def _apply_inverse_one_vector(self, v, mu=None, initial_guess=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        raise NotImplementedError

    def _apply_inverse_adjoint_one_vector(self, u, mu=None, initial_guess=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def apply(self, U, mu=None):
        if U.ndim == 1:
            U = U.reshape((1, -1))
        assert U.shape[1] == self.dim_source
        if len(U) == 0:
            return np.zeros((0, self.dim_range))
        data = self._prepare_apply(U, mu, 'apply')
        V = [self._apply_one_vector(u, mu=mu, prepare_data=data) for u in U]
        return np.array(V)

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        if V.ndim == 1:
            V = V.reshape((1, -1))
        assert V.shape[1] == self.dim_range
        if len(V) == 0:
            return np.zeros((0, self.dim_source))
        if initial_guess is not None and initial_guess.ndim == 1:
            initial_guess = initial_guess.reshape((1, -1))
        assert initial_guess is None or initial_guess.shape[1] == self.dim_source and len(initial_guess) == len(V)
        try:
            data = self._prepare_apply(V, mu, 'apply_inverse', least_squares=least_squares)
            U = [self._apply_inverse_one_vector(v, mu=mu,
                                                initial_guess=(initial_guess[i]
                                                               if initial_guess is not None else None),
                                                least_squares=least_squares, prepare_data=data)
                 for i, v in enumerate(V)]
        except NotImplementedError:
            return super().apply_inverse(V, mu=mu, least_squares=least_squares)
        return np.array(U)

    def apply_adjoint(self, V, mu=None):
        if V.ndim == 1:
            V = V.reshape((1, -1))
        assert V.shape[1] == self.dim_range
        if len(V) == 0:
            return np.zeros((0, self.dim_source))
        try:
            data = self._prepare_apply(V, mu, 'apply_adjoint')
            U = [self._apply_adjoint_one_vector(v, mu=mu, prepare_data=data) for v in V]
        except NotImplementedError:
            return super().apply_adjoint(V, mu=mu)
        return np.array(U)

    def apply_inverse_adjoint(self, U, mu=None, initial_guess=None, least_squares=False):
        raise NotImplementedError
        assert U in self.source
        try:
            data = self._prepare_apply(U, mu, 'apply_inverse_adjoint', least_squares=least_squares)
            V = [self._apply_inverse_adjoint_one_vector(u, mu=mu,
                                                        initial_guess=(initial_guess.vectors[i]
                                                                       if initial_guess is not None else None),
                                                        least_squares=least_squares, prepare_data=data)
                 for i, u in enumerate(U.vectors)]
        except NotImplementedError:
            return super().apply_inverse_adjoint(U, mu=mu, least_squares=least_squares)
        return self.range.make_array(V)


class LinearComplexifiedListVectorArrayOperatorBase(ListVectorArrayOperatorBase):
    """Base |Operator| for complexified |ListVectorArrays|."""

    linear = True

    @abstractmethod
    def _real_apply_one_vector(self, u, mu=None, prepare_data=None):
        pass

    def _real_apply_inverse_one_vector(self, v, mu=None, initial_guess=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def _real_apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        raise NotImplementedError

    def _real_apply_inverse_adjoint_one_vector(self, u, mu=None, initial_guess=None, least_squares=False,
                                               prepare_data=None):
        raise NotImplementedError

    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        real_part = self._real_apply_one_vector(u if np.isrealobj(u) else u.real, mu=mu, prepare_data=prepare_data)
        if not np.isrealobj(u):
            imag_part = self._real_apply_one_vector(u.imag, mu=mu, prepare_data=prepare_data)
            return real_part + 1j*imag_part
        else:
            return real_part

    def _apply_inverse_one_vector(self, v, mu=None, initial_guess=None, least_squares=False, prepare_data=None):
        real_part = self._real_apply_inverse_one_vector(v if np.isrealobj(v) else v.real, mu=mu,
                                                        initial_guess=((initial_guess if np.isrealobj(initial_guess)
                                                                        else initial_guess.real_part)
                                                                       if initial_guess is not None else None),
                                                        least_squares=least_squares,
                                                        prepare_data=prepare_data)
        if not np.isrealobj(v):
            imag_part = self._real_apply_inverse_one_vector(v.imag, mu=mu,
                                                            initial_guess=(initial_guess.imag
                                                                           if initial_guess is not None else None),
                                                            least_squares=least_squares,
                                                            prepare_data=prepare_data)
            return real_part + 1j*imag_part
        else:
            return real_part

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        raise NotImplementedError
        real_part = self._real_apply_adjoint_one_vector(v.real_part, mu=mu, prepare_data=prepare_data)
        if v.imag_part is not None:
            imag_part = self._real_apply_adjoint_one_vector(v.imag_part, mu=mu, prepare_data=prepare_data)
        else:
            imag_part = None
        return self.source.vector_type(real_part, imag_part)

    def _apply_inverse_adjoint_one_vector(self, u, mu=None, initial_guess=None, least_squares=False, prepare_data=None):
        real_part = self._real_apply_inverse_adjoint_one_vector(u.real_part, mu=mu,
                                                                initial_guess=(initial_guess.real_part
                                                                               if initial_guess is not None else None),
                                                                least_squares=least_squares,
                                                                prepare_data=prepare_data)
        if u.imag_part is not None:
            imag_part = self._real_apply_inverse_adjoint_one_vector(u.imag_part, mu=mu,
                                                                    initial_guess=(initial_guess.imag_part
                                                                                   if initial_guess is not None
                                                                                   else None),
                                                                    least_squares=least_squares,
                                                                    prepare_data=prepare_data)
        else:
            imag_part = None
        return self.range.vector_type(real_part, imag_part)
