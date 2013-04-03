# pymor (http://www.pymor.org)
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from .interfaces import DiscreteOperatorInterface, LinearDiscreteOperatorInterface
from .basic import GenericOperator, GenericLinearOperator
from .affine import LinearAffinelyDecomposedOperator
from .constructions import (ProjectedOperator, ProjectedLinearOperator, project_operator,
                            SumOperator, LinearSumOperator, add_operators)
