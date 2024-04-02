## L2loglink as a general loss if *y*≥0, with applications to ranking

### L2loglink as a general loss if *y*≥0

``` param = HTBoost(loss=:L2loglink) ```

The L2loglink loss is designed as a robust and general alternative to :gamma and other distributions
defined on y>0. When y is continuous, y = 0 is allowed in the :L2loglink function, but not in :gamma.
The L2loglink can also work well for count data, rank data, and any situations where y≥0.

The :L2loglink is a L2 loss, but with a log-link function, meaning that the tree ensemble approximates log(E(y|x)) (as standard, for example, for a gamma loss), so
loss = [y - exp(γ)]², and therefore γ = log(E(y|x)).
Compared to specialized distributions like the gamma, it sacrifices some efficiency if the assumed distribution is
indeed the true distribution, but can be more efficient if it is not.

I don't believe that this option is available in other GBM packages. XGBoost offers reg:squaredlogerror, which is 
a L2 loss on log(y+1). A similar option is also available in HTBoost, as loss = :lognormal, which is equivalent
to fitting a L2 (Gaussian) loss to log(y). The :L2loglink is different, as it fits a L2 loss to y, but with a log-link function.
The reason to specify these two separate options is that a lognormal is consistent for E(log(y)|x),
but not for necessarily E(y|x); the possible adjustment E(y|x) = exp(E(log(y)|x) + 0.5*Var(log(y)|x)) requires fitting a
separate model for Var(log(y)|x), or hoping that the variance is constant. On the other hand, :L2loglink is consistent for E(y|x).

See [examples/Gamma.jl](../../../examples/Gamma.jl) for a case where :L2loglink approaches the performance of :gamma even when the
data is generated from a gamma distribution.

### L2loglink loss as an interesting option for ranking problems.

HTBoost does not yet have specialized ranking losses. 

According to the benchmarks posted in [CatBoost ranking benchmarks](https://github.com/catboost/benchmarks/blob/master/ranking/Readme.md#4-results)
the :L2 loss can be surprisingly competitive in ranking tasks.
The :L2loglink loss available in HTBoost may be a better choice than :L2 for ranking tasks, since it enforces E(y|x)>0. 

The suggested practice is to make sure that ranking is expressed numerically, with *y* ∈ {1,2,...}, (NOTE: y=0 is allowed but best avoided in this case) and loss=:L2loglink. 