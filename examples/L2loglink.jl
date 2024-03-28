#= 

The :L2loglink loss is designed as a robust and general alternative to :gamma and other continuous distributions
defined on y>0. When y is continuous, y = 0 is allowed in the :L2loglink function, but not in :gamma.
The :L2loglink can also work well for for count data, rank data, and any situations where y≥0.

The :L2loglink is a L2 loss, but with a log-link function, meaning that the tree ensemble approximates log(E(y|x)).
loss = [y - exp(γ)]², where γ = log(E(y|x)).
Compared to specialized distributions like :gamma, it sacrifices some efficiency if the assumed distribution is
indeed the true distribution, but it is more robust to misspecification of the distribution.

I don't believe that this option is available in other GBM packages. XGBoost offers reg:squaredlogerror, which is 
a L2 loss on log(y+1). A similar option is also available in HTBoost, as loss = :lognormal, and is equivalent
to fitting a L2 (Gaussian) loss to log(y). The :L2loglink is different, as it fits a L2 loss to y, but with a log-link function.
The reason to specify these two separate options is that a lognormal is consistent for E(log(y)|x),
but not for necessarily E(y|x): the adjustment E(y|x) = exp(E(log(y)|x) + 0.5*Var(log(y)|x)) requires fitting a
separate model for Var(log(y)|x), or hoping that the variance is constant. On the other hand, :L2loglink is consistent for E(y|x).

See examples/Gamma.jl for a case where :L2loglink approaches the performance of :gamma even when the
data is generated from a gamma distribution.

See examples/Ranking.jl for why :L2loglink may often be a solid choice for ranking problems.

=#