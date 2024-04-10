## Missing data 

HTBoost should outperform other GBMs in the automatic treatment of missing values in *x*.  

**HTBoost handles missing values automatically. No user intervention required.** 

- data.x may contain NaN or *missing*. Missing values of *y* are discarded.
- Forecasting with missing values is also supported.
- HTBoost handles missing values (NaN or missing) internally, like other GBMs. The approach is Block Propagation (see Josse et al. 2020, *"On the consistency of supervised learning with missing values"*), as LightGBM, but with a key difference ...
- Taking advantage of soft splits (where splits are indeed soft), missing can be optimal allocated to either branch in fractional proportions, where the fraction is optimized at each split. The result is more efficient inference with missing values in finite samples.
- Unlike imputation, the internal assignments in all these GMBs recovers f(*x*) asymptotically whether data are missing at random, or missing not at random as a function of *x* only, or missing not at random as a function of E(*y*). (Josse et al. 2020)
-  When feasible, a high-quality imputation of missing values + mask may perform better, particularly in small samples, high predictability of missing values from non-missing values, linear or quasi-linear f(x), and missing at random (in line with the results of Josse et al.)  

There is an option *HTBparam(delete_missing=true)* to delete all rows with missing values, but, in light of the discussion above, this is not recommended. 

See [Missing.jl](../examples/Missing_data.jl) for a more detailed discussion and code to reproduce the simulations in Josse et al. 2020, including a comparison with LightGBM.
HTBoost strongly outperforms LightGBM in the settings of Josse et al. 2020.

