# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.rules import RuleTable, match_class, match_generic
from pymor.core.exceptions import ImageCollectionError, NoMatchingRuleError
from pymor.core.logger import getLogger
from pymor.operators.constructions import ConcatenationOperator, LincombOperator, SelectionOperator
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.operators.interface import Operator


def estimate_image(operators=(), vectors=(),
                   domain=None, extends=False, orthonormalize=True, product=None,
                   riesz_representatives=False):
    """Estimate the image of given |Operators| for all mu.

    Let `operators` be a list of |Operators| with common source and range, and let
    `vectors` be a list of |VectorArrays| or vector-like |Operators| in the range
    of these operators. Given a |VectorArray| `domain` of vectors in the source of the
    operators, this algorithms determines a |VectorArray| `image` of range vectors
    such that the linear span of `image` contains:

    - `op.apply(U, mu=mu)` for all operators `op` in `operators`, for all possible |Parameters|
      `mu` and for all |VectorArrays| `U` contained in the linear span of `domain`,
    - `U` for all |VectorArrays| in `vectors`,
    - `v.as_range_array(mu)` for all |Operators| in `vectors` and all possible |Parameters| `mu`.

    The algorithm will try to choose `image` as small as possible. However, no optimality
    is guaranteed. The image estimation algorithm is specified by :class:`CollectOperatorRangeRules`
    and :class:`CollectVectorRangeRules`.

    Parameters
    ----------
    operators
        See above.
    vectors
        See above.
    domain
        See above. If `None`, an empty `domain` |VectorArray| is assumed.
    extends
        For some operators, e.g. |EmpiricalInterpolatedOperator|, as well as for all
        elements of `vectors`, `image` is estimated independently from the choice of
        `domain`.  If `extends` is `True`, such operators are ignored. (This is useful
        in case these vectors have already been obtained by earlier calls to this
        function.)
    orthonormalize
        Compute an orthonormal basis for the linear span of `image` using the
        :func:`~pymor.algorithms.gram_schmidt.gram_schmidt` algorithm.
    product
        Inner product |Operator| w.r.t. which to orthonormalize.
    riesz_representatives
        If `True`, compute Riesz representatives of the vectors in `image` before
        orthonormalizing (useful for dual norm computation when the range of the
        `operators` is a dual space).

    Returns
    -------
    The |VectorArray| `image`.

    Raises
    ------
    ImageCollectionError
        Is raised when for a given |Operator| no image estimate is possible.
    """
    assert operators or vectors
    dim_domain = operators[0].dim_source if operators else None
    dim_image = operators[0].dim_range if operators \
        else vectors[0].shape[1] if isinstance(vectors[0], np.ndarray) \
        else vectors[0].dim_range
    assert all(op.dim_source == dim_domain and op.dim_range == dim_image for op in operators)
    assert all(
        isinstance(v, np.ndarray) and (
            v.shape[1] == dim_image
        )
        or isinstance(v, Operator) and (
            v.dim_range == dim_image and v.linear
        )
        for v in vectors
    )
    if domain is not None and domain.ndim == 1:
        domain = domain.reshape((1, -1))
    assert domain is None or dim_domain is None or domain.shape[1] == dim_domain
    assert product is None or product.dim_source == product.dim_range == dim_image

    image = np.zeros((0, dim_image))
    if not extends:
        rules = CollectVectorRangeRules(image)
        for v in vectors:
            try:
                rules.apply(v)
            except NoMatchingRuleError as e:
                raise ImageCollectionError(e.obj) from e
        image = rules.image

    if operators and domain is None:
        domain = np.zeros((0, dim_domain))
    for op in operators:
        rules = CollectOperatorRangeRules(domain, image, extends)
        try:
            rules.apply(op)
        except NoMatchingRuleError as e:
            raise ImageCollectionError(e.obj) from e
    image = rules.image

    if riesz_representatives and product:
        image = product.apply_inverse(image)

    if orthonormalize:
        image = gram_schmidt(image, product=product)

    return image


def estimate_image_hierarchical(operators=(), vectors=(), domain=None, extends=None,
                                orthonormalize=True, product=None, riesz_representatives=False):
    """Estimate the image of given |Operators| for all mu.

    This is an extended version of :func:`estimate_image`, which calls
    :func:`estimate_image` individually for each vector of `domain`.

    As a result, the vectors in the returned `image` |VectorArray| will
    be ordered by the `domain` vector they correspond to (starting with
    vectors which correspond to the elements of `vectors` and to |Operators|
    for which the image is estimated independently from `domain`).

    This function also returns an `image_dims` list, such that the first
    `image_dims[i+1]` vectors of `image` correspond to the first `i`
    vectors of `domain` (the first `image_dims[0]` vectors correspond
    to `vectors` and to |Operators| with fixed image estimate).

    Parameters
    ----------
    operators
        See :func:`estimate_image`.
    vectors
        See :func:`estimate_image`.
    domain
        See :func:`estimate_image`.
    extends
        When additional vectors have been appended to the `domain` |VectorArray|
        after :func:`estimate_image_hierarchical` has been called, and
        :func:`estimate_image_hierarchical` shall be called again for the extended
        `domain` array, `extends` can be set to `(image, image_dims)`, where
        `image`, `image_dims` are the return values of the last
        :func:`estimate_image_hierarchical` call. The old `domain` vectors will
        then be skipped during computation and `image`, `image_dims` will be
        modified in-place.
    orthonormalize
        See :func:`estimate_image`.
    product
        See :func:`estimate_image`.
    riesz_representatives
        See :func:`estimate_image`.

    Returns
    -------
    image
        See above.
    image_dims
        See above.

    Raises
    ------
    ImageCollectionError
        Is raised when for a given |Operator| no image estimate is possible.
    """
    assert operators or vectors
    dim_domain = operators[0].dim_source if operators else None
    dim_image = operators[0].dim_range if operators \
        else vectors[0].shape[1] if isinstance(vectors[0], np.ndarray) \
        else vectors[0].dim_range
    assert all(op.dim_source == dim_domain and op.dim_range == dim_image for op in operators)
    assert all(
        isinstance(v, np.ndarray) and (
            v.shape[1] in dim_image
        )
        or isinstance(v, Operator) and (
            v.dim_range == dim_image and v.linear
        )
        for v in vectors
    )
    assert domain is None or dim_domain is None or domain.shape[1] == dim_domain
    assert product is None or product.dim_source == product.dim_range == dim_image
    assert extends is None or len(extends) == 2

    logger = getLogger('pymor.algorithms.image.estimate_image_hierarchical')

    if operators and domain is None:
        domain = np.zeros((0, dim_domain))

    if extends:
        image = extends[0]
        image_dims = extends[1]
        ind_range = range(len(image_dims) - 1, len(domain)) if operators else range(len(image_dims) - 1, 0)
    else:
        image = np.zeros((0, dim_image))
        image_dims = []
        ind_range = range(-1, len(domain)) if operators else [-1]

    for i in ind_range:
        logger.info(f'Estimating image for basis vector {i} ...')
        if i == -1:
            new_image = estimate_image(operators, vectors, None, extends=False,
                                       orthonormalize=False, product=product,
                                       riesz_representatives=riesz_representatives)
        else:
            new_image = estimate_image(operators, [], domain[i], extends=True,
                                       orthonormalize=False, product=product,
                                       riesz_representatives=riesz_representatives)

        gram_schmidt_offset = len(image)
        image = np.vstack([image, new_image])
        if orthonormalize:
            with logger.block('Orthonormalizing ...'):
                image = gram_schmidt(image, offset=gram_schmidt_offset, product=product)
            image_dims.append(len(image))

    return image, image_dims


class CollectOperatorRangeRules(RuleTable):
    """|RuleTable| for the :func:`estimate_image` algorithm."""

    def __init__(self, source, image, extends):
        super().__init__(use_caching=True)
        self.__auto_init(locals())

    @match_generic(lambda op: op.linear and not op.parametric)
    def action_apply_operator(self, op):
        self.image = np.vstack([self.image, op.apply(self.source)])

    @match_class(LincombOperator, SelectionOperator)
    def action_recurse(self, op):
        self.apply_children(op)

    @match_class(EmpiricalInterpolatedOperator)
    def action_EmpiricalInterpolatedOperator(self, op):
        if hasattr(op, 'collateral_basis') and not self.extends:
            self.image = np.vstack([self.image, op.collateral_basis])

    @match_class(ConcatenationOperator)
    def action_ConcatenationOperator(self, op):
        if len(op.operators) == 1:
            self.apply(op.operators[0])
        else:
            firstrange = np.zeros((0, op.operators[-1].dim_range))
            rules = type(self)(self.source, firstrange, self.extends)
            rules.apply(op.operators[-1])
            first_range = rules.image
            rules = type(self)(firstrange, self.image, self.extends)
            rules.apply(op.with_(operators=op.operators[:-1]))
            self.image = rules.image


class CollectVectorRangeRules(RuleTable):
    """|RuleTable| for the :func:`estimate_image` algorithm."""

    def __init__(self, image):
        super().__init__(use_caching=True)
        self.image = image

    @match_class(np.ndarray)
    def action_VectorArray(self, obj):
        self.image = np.vstack([self.image, obj])

    @match_generic(lambda op: op.linear and not op.parametric)
    def action_as_range_array(self, op):
        self.image = np.vstack([self.image, op.as_range_array()])

    @match_class(LincombOperator, SelectionOperator)
    def action_recurse(self, op):
        self.apply_children(op)

    @match_class(ConcatenationOperator)
    def action_ConcatenationOperator(self, op):
        if len(op.operators) == 1:
            self.apply(op.operators[0])
        else:
            firstrange = op.operators[-1].range.empty()
            rules = CollectVectorRangeRules(firstrange)
            rules.apply(op.operators[-1])
            firstrange = rules.image
            rules = CollectOperatorRangeRules(firstrange, self.image, False)
            rules.apply(op.with_(operators=op.operators[:-1]))
            self.image = rules.image
