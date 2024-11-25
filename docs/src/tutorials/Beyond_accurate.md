# Beyond accurate

Modality = :accurate should cover the needs of most users. 

In situations where computing time is not a factor and even the smallest increment in performance matter, the following may be tried:

- Lower lambda to 0.05, particularly if the function is highly nonlinear (some features have high average τ values).
  In some cases this may require increasing the maximum number of trees, e.g HTBparam(ntrees=5000).

- HTBboost cross-validates depth up to 6. This is sufficient in most circustances. If the best depth is 6, try 7 and perhaps even 8. (Note that computing time can easily double with each increment in depth.)
  This can be achieved by running, for example: 
```julia  
  output = HTBfit(data,param,cv_grid=[5,6,7,8])
```
- Consider alternative loss functions, such as :L2loglink or (particularly for small n) :t.    

- If the signal-to-noise ratio is very low, set cv_sparsity = true. (The default sets it to true only
  if n/p is not large, but very noisy data may benefit from it.)
```julia  
  output = HTBfit(data,param,cv_sparsity=true)
```
In datasets where the accuracy of HTBoost is roughly comparable to that of XGBoost and LightGBM,
a 50-50 combination (or more sophisticated stacking) of HTBoost with one of the other two is likely 
to yield the best results. (If HTBoost is mostly selecting sharp splits ---  as illustrated in [Basic use](Basic_use.md) -- it will however be much faster to combine XGBoost with CatBoost rather than with HTBoost).  
