## HTBoost Tutorials

The following tutorials cover hands-on use of HTBoost.
The [examples](../../examples/examples.md) provide more cases of different loss functions as well as illustrations to understand what HTBoost does and how it differs from other GBMs.

### Most important user cases 

  * [Basic use](./tutorials/Basic%20use.md) (main options, cv, savings and loading results, variable importance and more post-estimation analysis)
  * [Logistic regression](tutorials/Logistic.md) (binary classification; comparison with LightGBM)
  * [Time series and panels](tutorials/Time%20series%20and%20panels.md) (Data Frames, time series and panels/longitudinal data, with various options for cv)
  * [Categorical features](tutorials/Categoricals.md) (how HTBoost handles categorical features; comparison with LightGBM and CatBoost)
  * [Missing data](tutorials/Missing.md) (HTBoost excels at handling missing data)
  * [Speeding up HTBoost with large n](tutorials/Faster%20large%20n.md) (strategies to reduce computing time for large n)


### Other distributions (loss functions)

  * [Really robust regression](tutorials/t.md) (student-t and Huber, done right; comparison with LightGBM)
  * [Multiclass](tutorials/Multiclass.md) (multiclass classification; comparison with LightGBM)
  * [Zero inflated y](tutorials/Zero%20inflated%20y.md) (loss functions for zero-inflated *y*; comparison with LightGBM)
  * [Poisson and GammaPoisson](tutorials/GammaPoisson.md) (aka negative binomial for count data; comparison with LightGBM)  
  * [L2loglink and rank](tutorials/L2loglink%20and%20rank.md) (a new option if min(y)≥0, whether continuous, count, or rank)

### Others

  * [Offset (exposure)](tutorials/Offset.jl) (how to add an offset, common in e.g. insurance, biology ...)
  * [Beyond accurate](tutorials/Beyond%20accurate.md) (some suggestions when chasing maximum accuracy)
  

