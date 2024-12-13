# When is HTBoost likely to outperform (underperform) other GBMs like XGB and LightGBM?

## When is HTBoost more likely to outperform? 

When one or more of the following conditions are met:

  * The underlying function is smooth with respect to at least a subset of the features. This can be assessed and visualized using `HTBweightedtau()` (see [Basic use](tutorials/Basic_use.md))
  * Small, highly unbalanced, or noisy datasets.

## What to do when HTBoost performs approximately as well as other GBM.

  * If time is of the essence and/or accuracy and data efficiency are not priorities, drop HTBoost.
  * If maximizing accuracy is important, then combinations (stacking) of HTBoost with XGB and/or LightGMB typically improve on XGB/LightGBM (the different tree construction results in less than perfect correlation in predictions). However, if the function shows little smoothness (which can be assessed and visualized using `HTBweightedtau()`, see [Basic use](tutorials/Basic_use.md)), CatBoost is a more computationally efficient option unless other positive features of HTBoost are relevant (see [index](index.md)) 
 

## When is HTBoost outperformed by other GBM?

Different packages handle categorical features differently in their default mode, which can lead to different performance if categorical features are prominent. Other than that, if HTBoost is fitted in modality=:accurate or :compromise, it may slightly underperform in situations where symmetric trees are inferior to non-symmetric trees or when the underlying function is so irregular that the preliminary optimization (based on a rough grid) finds a local mode and does not split on the best feature. It will also slightly underperform is very deep trees (depth >>7) give the best fit. These conditions are probably more likely with near-perfect fit. The first condition can be evaluated by running CatBoost with `grow_policy = SymmetricTree` and `grow_policy = Depthwise`, and comparing the results. The second problem would be solved by setting param.mugridpoints >> 10, but computing times would increase proportionally. Finally, HTBoost becomes too slow to run with depth larger than 6 or 7; if the setting requires deeper trees, performance may suffer.  

