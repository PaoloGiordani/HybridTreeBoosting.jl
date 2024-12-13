
Note: all Julia symbols can be replaced by strings. e.g. loss=:L2 can be replaced by loss="L2".

## Parameters more likely to be modified by user

- `loss`             [:L2] Supported distributions:
    - :L2 (Gaussian)
    - :logistic (binary classification)
    - :multiclass (multiclass classification)
    - :t (student-t, robust alternative to :L2)
    - :Huber 
    - :gamma 
    - :Poisson (count data)
    - :gammaPoisson (aka negative binomial, count data)
    - :L2loglink (alternative to :L2 if *y*≥0)
    - :lognormal (L2 on log(*y*))
    - :hurdleGamma (zero-inflated *y*)
    - :hurdleL2loglink (zero-inflated *y*)
    - :hurdleL2 (zero-inflated *y*)

- See the [examples](Examples.md) and [tutorials](Tutorials.md) for uses of each loss function. Fixed coefficients (such as shape for :gamma, dispersion and dof for :t, and overdispersion for :gammaPoisson) are computed internally by maximum likelihood. Inspect them using *HTBcoeff()*.
In *HTBpredict()*, predictions are for E(*y*) if predict=:Ey (default), while predict=:Egamma forecasts the fitted parameter ( *E(logit(prob)* for :logistic, *log(E(y))* for :gamma etc ... )

- `modality`         [:compromise]  Options are: :accurate, :compromise (default), :fast, :fastest.  
                     :fast and :fastest run only one model, while :compromise and :accurate cross-validate the most important parameters.
                     :fast runs only one model (only cv number of trees) at values defined in param = *HTBparam()*. 
                     :fastest runs only one model, setting lambda=0.2, nfold=1 and nofullsample=true (does not re-estimate on the full sample after cv).
                      Recommended for faster preliminary analysis only.
                      In most cases, :fast and :fastest also use the quadratic approximation to the loss for large samples.
                      :accurate cross-validates 7-10 models (at the most important parameters (see HTBfit() for details),
                      then stacks all the cv models. :compromise cv 4-7 models.  
                                        
- `randomizecv`       [false] default is block-cv (aka purged cv); a time series or panel structure is automatically detected (see *HTBdata()*)
                            if a date column is provided. Set to true for standard cv.

- `nfold`              [4] n in n-fold cv. Set nfold = 1 for a single validation set (by default the last param.sharevalidation share of the sample).
                            nfold, sharevalidation, and randomizecv are disregarded if train and test observations are provided by the user.

- `sharevalidation:`        [0.30] Can be: a) Integer, size of the validation set, or b) Float, share of validation set.
                            Relevant only if nfold = 1. If randomizecv = false (default), the validation set is compromised of the last x% observations, else a random subsample. 

- `indtrain_a:Vector{Vector{I}} ` [ ] for user's provided array of indices of train sets. e.g. vector of 5 vectors, each with indices of train set observations

- `indtest_a:Vector{Vector{I}} `  [ ] for user's provided array of indices of test sets. e.g. vector of 5 vectors, each with indices of train set observations. 

- `nofullsample`      [false] if true and nfold=1, HTBoost is not re-estimated on the full sample after validation.
                            Reduces computing time by roughly 60%, at the cost of a modest loss of accuracy.
                            Useful for very large datasets, in preliminary analysis, in simulations, and when instructions specify a train/validation
                            split with no re-estimation on full sample. Activated by default when modality=:fastest.     

- `cat_features`            [ ] vector of indices of categorical features, e.g. [2,5], or vector of names in DataFrame,
                            e.g. [:wage,:age] or ["wage","age"]. If empty, categoricals are automatically detected as non-numerical features.

- `overlap:`            [0] number of overlaps in time series and panels. Typically overlap = h-1, where *y(t) = Y(t+h)-Y(t)*. Used for purged-cv.

- `verbose`         [:Off] verbosity :On or :Off

- `warnings`        [:On] or :Off


## Parameters less frequently modified by user

- `priortype`               [:hybrid] Options are: :hybrid, :smooth, :disperse.  
:hybrid encourages smoothness, but allows both smooth and sharp splits, :smooth forces smooth splits,
                            :disperse is :hybrid but with no penalization encouraging smooth functions (not recommended in most cases).
                            Set to :smooth if you want to force derivatives to be defined everywhere, but note that this disengages hybrid trees and can lead to substantial loss of accuracy if the function is not smooth everywhere. 

- `lambda`           [0.1 or 0.2] Learning rate. 0.1 for (nearly) best performance. 0.2 can be almost as accurate, particularly if the function is smooth and p is small.
                     The default is 0.1, except in modality = :fastest, where it's 0.2.
                     Consider 0.05 if tiny improvements in accuracy are important and computing time is not a concern (and possibly increase ntrees to 4000)

- `depth`              [5] tree depth. Unless modality = :fast or :fastest, this is over-written as depth is cross-validated. See *HTBfit()* for more options.

- `weights`                 NOTE: weights for weighted likelihood are set in *HTBdata()*, not in *HTBparam()*.
- `offset`                  NOTE: offsets (aka exposures) are set in *HTBdata()*, not in *HTBparam()*. See [offset tutorial](tutorials/Offset.md) and [offset example](examples/Offset.md)    

- `sparsity_penalization`   [0.3] positive numbers encourage sparsity. The range [0.0-1.5] should cover most scenarios. 
                            Automatically cv in modality=:compromise and :accurate. Increase to obtain a more parsimonious model, set to 0 for standard boosting.

- `depth`              [5] tree depth. Unless modality = :fast or :fastest, this is over-written as depth is cross-validated. See *HTBfit()* for more options.
- `ntrees`             [2000] Maximum number of trees. *HTBfit()* will automatically stop when cv loss stops decreasing.

- `sharevs`                 [1.0] row subsampling in variable selection phase (only to choose feature on which to split.) Default is no subsampling.
                            sharevs = :Auto sets the subsample size to min(n,50k*sqrt(n/50k)).
                            At high n, sharevs<1 speeds up computations, but can reduce accuracy, particularly in sparse setting with low SNR.         

- `subsampleshare_columns`  [1.0] column subsampling (aka feature subsampling) by tree.

- `min_unique`              [:default] sharp splits are imposed on features with less than min_unique values (default is 5 for modality=:compromise or :accurate, else 10)

- `mixed_dc_sharp`          [false] true to force sharp splits on discrete and mixed discrete-continuous features (defined as having over 20% obs on a single value)

- `delete_missing`          [false] true to delete rows with missing values in any feature, false to handle missing internally (recommended).

- `theta`                   [1]  numbers larger than 1 imply tighter penalization on β (final leaf values) compared to default.

- `meanlntau`        [1.0] prior mean of log(τ). Set to higher numbers to suggest less smooth functions.        

- `mugridpoints`       [11] number of points at which to evaluate μ during variable selection. 5 is sufficient on simulated data with normal or uniform distributions, but actual data may benefit from more (due to with highly non-Gaussian features).
                            For extremely complex and nonlinear features, more than 10 may be needed. This number is automatically reduced at deep levels of the tree (higher than 5 in default).       

- `force_sharp_splits`      [ ] optionally, a p vector of Bool, with j-th value set to true if the j-th feature is forced to enter with a sharp split.

- `force_smooth_splits`     [ ] optionally, a p vector of Bool, with j-th value set to true if the j-th feature is forced to enter with a smooth split (high values of τ not allowed).

- `cat_representation_dimension`  [3] 1 for mean encoding, 2 also adds frequency, 3 also adds variance.

- `losscv`                  [:default] loss function for cross-validation. The default is the same loss type used for the training set. Other options are (:mse,:mae,:logistic,:sign).

- `n_refineOptim`      [10^6] maximum number of observations to use fit μ and τ (split point and smoothness).
                            Lower numbers can provide speed-ups with very large n at some cost in terms of fit.

- `loglikdivide`         [1.0] Higher numbers increase the strength or all priors. The defaults sets it internally using *HTBloglikdivide()*,
                            when it detects a dates series in *HTBdata()*.

- `tau_threshold`         [10.0] lowest threshold for imposing sharp splits. Lower numbers give more sharp splits.

- `multiplier_stdtau`    [5.0] The default priors suggest smoother splits on features whose unconditional distribution (appropriately transformed according to the link function) is closer to the unconditional distribution of *y* or, when not applicable, to a Gaussian. To disengage this feature, set *multiplier_stdtau* = 0

Additional parameters to control the parameter tuning process can be set in *HTBfit()*, but keeping the defaults is generally encouraged.


