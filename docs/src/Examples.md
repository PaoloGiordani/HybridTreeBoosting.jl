## Learning HTBoost via examples

The examples are Julia scripts that you can run. Some as similar to the tutorials, others explore additional aspects of HTBoost.

A good way to familiarize yourself with HTBoost and compare (its performance to LigthGBM) is to study and run the following examples:
  
  * [Basic use](examples/Basic_use.jl) (main options, cv, savings and loading results, variable importance and more post-estimation analysis)
  * [Logistic](examples/Logistic.jl) (binary classification)
  * [Global Equity Panel](examples/Global_Equity_Panel.jl) (time series and panels/longitudinal data, with various options for cv)
  * [Categoricals](examples/Categoricals.jl) (how HTBoost handles categorical features)
  * [Missing data](examples/Missing_data.jl)  (HTBoost excels at handling missing data)
  * [Speeding up with large n](examples/Speeding_up_with_large_n.jl) (strategies to reduce computing time for large n)

The other examples explore more specific aspects of HTBoost: 

Understanding hybrid trees 
  * [Hybrid trees](examples/Hybrid_trees.jl) (how HTBoost can escape local minima of smoothtrees)
  * [Projection pursuit regression](examples/Projection_pursuit_regression.jl) (an example where adding a single index model to each tree (the default in HTBoost) improves forecasting)

Other distributions (loss functions)
  * [Multiclass](examples/Multiclass.jl) (multiclass classification)
  * [Zero inflated y](examples/Zero_inflated_y.jl) (y≥0, continuous except for positive mass at 0)
  * [GammaPoisson](examples/GammaPoisson.jl) (aka negative binomial for count data)  
  * [Huber and t unbiased](examples/Huber_and_t_unbiased.jl) (outlier robust losses in HTBoost and lightGBM)
  * [t distribution](examples/t.jl) (the recommended robust loss in HTBoost)
  * [Gamma distribution](examples/Gamma.jl) (discusses options if min(y)>0)

Others

  * [Offset (exposure)](examples/Offset.jl) (how to add an offset, common in e.g. insurance, biology ...)
  * [Sparsity penalization](examples/Sparsity_penalization.jl) (how HTBoost improves forecasting by feature selection when p is large)
  * [Speedups with sparsevs](examples/Speedups_with_sparsevs.jl) (how HTBoost speeds up feature selection when p is large)


