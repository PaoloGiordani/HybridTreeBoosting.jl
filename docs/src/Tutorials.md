## HTBoost Tutorials

The following tutorials cover hands-on use of HTBoost.
The [examples](../../examples/examples.md) provide more illustrations to understand what HTBoost does and how it differs from other GMBs.

  
  * [Basic use](./tutorials/Basic%20use.md) (main options, cv, savings and loading results, variable importance and more post-estimation analysis)
  * [Logistic regression](tutorials/Logistic.md) (binary classification; comparison with LightGBM)
  * [Time series and panels](tutorials/Time%20series%20and%20panels.md) (Data Frames, time series and panels/longitudinal data, with various options for cv)
  * [Categorical features](tutorials/Categoricals.md) (how HTBoost handles categorical features; comparison with LightGBM and CatBoost)
  * [Missing data](tutorials/Missing.md) (HTBoost excels at handling missing data)
  * [Speeding up HTBoost with large n](tutorials/Faster%20large%20n.md) (strategies to reduce computing time for large n)

---------------------------------------------------------------------------
  
The other examples explore more specific aspects of HTBoost: 

Understanding hybrid trees 
  * [Hybrid trees](tutorials/Hybrid%20trees.md) (how HTBoost can escape local minima of smoothtrees)
  * [Projection pursuit regression](tutorials/Projection%20pursuit%20regression.md) (an example where adding a single index model to each tree (the default in HTBoost) improves forecasting)

Other distributions (loss functions)
  * [Multiclass](tutorials/Multiclass.md) (multiclass classification)
  * [Zero inflated y](tutorials/Zero%20inflated%20y.md) (y≥0, continuous except for positive mass at 0)
  * [GammaPoisson](tutorials/GammaPoisson.md) (aka negative binomial for count data)  
  * [Huber and t unbiased]tutorials/Huber%20and%20t%20unbiased.md) (outlier robust losses in HTBoost and lightGBM)
  * [t distribution](tutorials/t.md) (the recommended robust loss in HTBoost)
  * [Gamma distribution](tutorials/Gamma.md) (discusses options if min(y)>0)
  * [L2loglink](tutorials/L2loglink.md) (discusses more options if min(y)≥0, whether continuous, count, or rank)
  * [Ranking](tutorials/Ranking.md) (discussion of options for ranking tasks)

Others

  * [Offset (exposure)](tutorials/Offset.md) (how to add an offset, common in e.g. insurance, biology ...)
  * [Beyond accurate](tutorials/Beyond%20accurate.md) (some suggestions when chasing maximum accuracy)
  * [Sparsity penalization](tutorials/Sparsity%20penalization.md) (how HTBoost improves forecasting by feature selection when p is large)
  * [Speedups with sparsevs](tutorials/Speedups%20with%20sparsevs.md) (how HTBoost speeds up feature selection when p is large)


