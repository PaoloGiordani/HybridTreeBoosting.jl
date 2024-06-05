## Learning HTBoost via examples

The examples are Julia scripts that you can run. Some as similar to the tutorials, others explore additional aspects of HTBoost.

A good way to familiarize yourself with HTBoost and compare (its performance to LigthGBM) is to study and run the following examples:
  
  * [Basic use](examples/Basic_use.md) (main options, cv, savings and loading results, variable importance and more post-estimation analysis)
  * [Logistic](examples/Logistic.md) (binary classification)
  * [Global Equity Panel](examples/Global_Equity_Panel.md) (time series and panels/longitudinal data, with various options for cv)
  * [Categoricals](examples/Categoricals.md) (how HTBoost handles categorical features)
  * [Missing data](examples/Missing_data.md)  (HTBoost excels at handling missing data)
  * [Speeding up with large n](examples/Speeding_up_with_large_n.md) (strategies to reduce computing time for large n)

The other examples explore more specific aspects of HTBoost: 

Understanding hybrid trees 
  * [Hybrid trees](examples/Hybrid_trees.md) (how HTBoost can escape local minima of smoothtrees)
  * [Projection pursuit regression](examples/Projection_pursuit_regression.md) (an example where adding a single index model to each tree (the default in HTBoost) improves forecasting)

Other distributions (loss functions)
  * [Multiclass](examples/Multiclass.md) (multiclass classification)
  * [Zero inflated y](examples/Zero_inflated_y.md) (y≥0, continuous except for positive mass at 0)
  * [GammaPoisson](examples/gammaPoisson.md) (aka negative binomial for count data)  
  * [Huber and t unbiased](examples/Huber_and_t_unbiased.md) (outlier robust losses in HTBoost and lightGBM)
  * [t distribution](examples/t.md) (the recommended robust loss in HTBoost)
  * [Gamma distribution](examples/Gamma.md) (discusses options if min(y)>0)

Others

  * [Offset (exposure)](examples/Offset.md) (how to add an offset, common in e.g. insurance, biology ...)
  * [Sparsity penalization](examples/Sparsity_penalization.md) (how HTBoost improves forecasting by feature selection when p is large)
  * [Speedups with sparsevs](examples/Speedups_with_sparsevs.md) (how HTBoost speeds up feature selection when p is large)


