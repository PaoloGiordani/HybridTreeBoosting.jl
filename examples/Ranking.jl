""" 

Using HTBoost to rank 

HTBoost does not yet have specialized ranking losses. 

According to the benchmarks posted in https://github.com/catboost/benchmarks/blob/master/ranking/Readme.md#4-results
the :L2 loss can be surprisingly competitive in ranking tasks.
The :L2loglink loss available in HTBoost may be a better choice than :L2 for ranking tasks, since it enforces E(y|x) > 0. 

""" 