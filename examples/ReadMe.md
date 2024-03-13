# Learning HTBoost via examples

A good way to familiarize yourself with HTBoost is to study the following examples:

  * `Basic use` (main options, cv, savings and loading results)
  * `Logistic` (binary classification)
  * `Global Equity Panel` (time series and panels/longitudinal data)
  * `Categoricals` (how HTBoost handles categorical features)
  * `Missing data` (HTBoost excels at handling missing data)
  * `Speeding up large sample size` (strategies to reduce computing time for large n)

The other examples explore more specific aspects of HTBoost: 

Other distributions (loss functions)
  * `Multiclass` (multiclass classification)
  * `Zero inflated y` (y≥0, continuous except for positive mass at 0)
  * `GammaPoisson` (aka negative binomial for count data)  
  * `Huber and t unbiased` (outlier robust losses in HTBoost and lightGBM)
  * `t distribution` (the recommended robust loss in HTBoost)
  * `gamma distribution` (discusses options if min(y)>0)
  * `Ranking` (discussion of options for ranking tasks)
  
Miscellanea

  * `Offset or exposure` (how to add an offset, common in e.g. insurance, biology ...)
  * `Sparsity penalization` (how HTBoost improves forecasting by feature selection when p is large)
  * `Projection pursuit regression` (an example where adding a single index model to each tree (the default in HTBoost) improves forecasting)
  * `Speedups with sparsevs` (how HTBoost speeds up feature selection when p is large)

# When is HTBoost likely to outperform (underperform) other GBM like XGB and LightGBM?

## When is HTBoost more likely to outperform? 

When one or more of the following conditions are met:

  * The underlying function is smooth, which can be evaluated and visualized using `HTBweightedtau()` (see Examples\Basic use)
  * Small or noisy datasets.

## What to do when HTBoost performs approximately as well as other GBM.

  * If time is of the essence and/or accuracy is not extremely important, drop HTBoost.
  * If maximizing accuracy is important, then combinations (stacking) of HTBoost with XGB and/or LightGMB often improve on XGB/LightGBM (the different tree constructions results in less than perfect correlation in predictions).

## When is HTBoost outperformed by other GBM?

If HTBoost is fitted in modality=:accurate or :compromise, our experience so far is that it underperform in situations where symmetric trees are inferior non-symmetric trees or when the underlying function is so irregular that the optimization finds a local mode. The first condition can be evaluated by running CatBoost with `grow_policy = SymmetricTree` and `grow_policy = Depthwise`, and comparing the results. The second problem would be solved by setting param.mugridpoints >> 10, but computing times would increase proportionally. Finally, HTBoost becomes too slow to run with depth larger than 6 or 7. If the setting requires deeper trees, performance may suffer (rare?).  

