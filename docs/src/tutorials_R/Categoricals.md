## Categoricals features

HTBoost is promising for categorical features, particularly if high dimensionals.  
This tutorials shows:
- How to inform HTBoost about categorical features.
- Parameters related to categorical features, and their default values.
- See [Categorical features](../tutorials/Categoricals.md) for a comparison with LightGBM and CatBoost, with discussion.  

### How to inform HTBoost about categorical features in R 

When y and/or x contain strings (categorical features), we must translate our R dataframe into a Julia DataFrame, which is then fed to HTBdata(), e.g. (continuing from the previous example)

```r
x_string =  sample(c("v1", "v2", "v3"), n, replace = TRUE)   # create a categorical with 3 values
df       = data.frame(x,x_string)                          # R dataframe 
x        = DataFrames$DataFrame(df)                          # x is a Julia dataframe
data     = HybridTreeBoosting$HTBdata(y,x,param,fnames=colnames(df))    # pass the column names 
output   = HybridTreeBoosting$HTBfit(data,param)                        

```

Columns of string values are automatically interpreted by HTBoost as a categorical. If some categorical features are represented by numerical values, it is necessary to list them in param (in which case all categorical features, even strings, must be listed). This can be done either with a vector of their column positions, or with their names, if fnames (an optional argument) is provided to HTBdata()

```r
# either 
param = HybridTreeBoosting$HTBparam(cat_features=c(3))
data  = HybridTreeBoosting$HTBdata(y,x,param)    # passing the column names is optional

# or
param = HybridTreeBoosting$HTBparam(cat_features=c("x_string"))
data  = HybridTreeBoosting$HTBdata(y,x,param,fnames=colnames(df))    # passing the column names is required

```

See [examples/Categoricals](../examples/Categoricals.md) for a discussion of how HTBoost treats categoricals under the hood. Key points:
- Missing values are assigned to a new category.
- If there are only 2 categories, a 0-1 dummy is created. For anything more than two categories, it uses a variation of target encoding.
- The categories are encoded by 4 values in default mode: mean, frequency, variance (robust) and skew(robust). (For financial variables, the variance and skew may be more informative than the mean.) Set cat_representation_dimension = 1 to encode by mean only.
- One-hot-encoding with more than 2 categories is not supported, but can of course be implemented as data preprocessing.

## Cross-validation of categorical parameters 

 `param$cv_categoricals` can be used to perform a rough cross-validation of `n0_cat` and/or `mean_encoding_penalization`, as follows:
 - `cv_categoricals = "none"` uses default parameters 
 - `cv_categoricals = "penalty"` runs a rough cv the penalty associated to the number of categories; recommended if n/n_cat if high for any feature, particularly if SNR is low                             
 - `cv_categoricals = "n0"` runs a rough of cv the strength of the prior shrinking categorical values to the overall mean; recommended with highly unequal number of observations in different categories.
- `cv_categoricals = "both"` runs a rough cv of penalty and n0 

The default is "none" if :modality in ("fastest","fast"), "penalty" if "compromise", and "both" if "accurate". 


### Comparison to LightGBM and CatBoost

Different packages differ substantially in their treatment of categorical features.  
LightGBM does not use target encoding, and can completely break down (very poor in-sample and oos fit) when the number of categories is high in relation to n (e.g. n=10k, #cat=1k). The LightGBM manual suggests
treating high dimensional categorical features as numerical or embedding them in a lower-dimensional space. LightGBM can, however, perform very well in lower-dimensional cases.

CatBoost, in contrast, adopts mean target encoding as default, can handle very high dimensionality and
has a sophisticated approach to avoiding data leakage which HTBoost is missing. (HTBoost resorts to a penalization on categorical features instead.) CatBoost also interacts categorical features, while HTBoost does not.
In spite of the less sophisticated treatment of categoricals, in this simple simulation set-up HTBoost substantially outperforms CatBoost if n_cat is high and the categorical feature interacts with the continuous feature,
presumably because target encoding generates smooth functions  by construction in this setting.

It seems reasonable to assume that high dimensional target encoding, by its very nature, will generate smooth functions in many settings, making 
HTBoost a promising tool for high dimensional categorical features. The current treatment of categorical features is however quite
crude compared to CatBoost, so some of these gains are not yet realized. 

See [Categorical features](../tutorials/Categoricals.md) for a comparison with LightGBM and CatBoost on simulated data, with discussion.
