""" 

The :L2loglink loss is designed as a robust and general alternative to :gamma and other continuous distributions
defined on y>0. When y is continuous, y = 0 is allowed in the :L2loglink function, but not in :gamma.
The :L2loglink can also work well for for count data, rank data, and any situations where y>0.

The :L2loglink is a L2 loss, but with a log-link function, meaning that the tree ensemble approximates log(E(y|x)).
Compared to specialized distributions like :gamma, it sacrifices some efficiency if the assumed distribution is
indeed the true distribution, but it is more robust if the true distribution is not the assumed one.

See examples/Gamma distribution.jl for a case where :L2loglink approaches the performance of :gamma even when the
data is generated from a :gamma distribution.

See examples/Ranking.jl for why :L2loglink may often be a solid choice for ranking problems.

"""