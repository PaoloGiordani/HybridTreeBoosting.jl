## Time series and panels

### Working with time series and longitudinal data (panels). Data as DataFrame.

The user only needs to provide features and, optionally, a vector of dates in *HTBdata( )* and, if there is overlapping, the overlap parameter in *HTBparam()*.
  Example:
```r 
param  = HTBoost$HTBparam(overlap=20)         
data   = HTBoost$HTBdata(y,x,param,dates,fnames = fnames)
```

There is no need to specify *dates* for an individual time series. For a panel, *dates* is required for correct default block-cv (but not if user-specified train and test samples are provided in indtrain_a and indtest_a).

Overlap defaults to 0. Typically overlap = h-1, where y(t) = Y(t+h)-Y(t) (e.g. Y a log price and h=20 for monthly financial returns from daily data). This is used for purged-CV and to calibrate the priors.

By default, HTBoost uses block-cv, which is suitable for time series and longitudinal data. 
Another good alternative for times series and panels is expanding window cross-validation, which requires the user to provide indtrain_a and indtest_a in *HTBparam( )*.
The function *HTBindexes_from_dates()* can assist in building indtrain_a and indtest_a.

## Preliminary steps required in all scripts 

See [Basic_use](Basic_use.md)

### Some user's inputs for this dataset

```r

# data
log_ret = FALSE    # TRUE to predict log returns, FALSE (default) to predict returns
overlap = 0        # 0 for non-overlapping (default), h-1 for overlapping data, where h is the forecast horizon. 

# HTBoost

loss = "L2"  # if log_ret=FALSE, consider "L2loglink" instead of "L2"
modality = "accurate"    # "accurate", "compromise", "fast", "fastest"
priortype = "hybrid"     # "hybrid" (accurate), "smooth" (forces smooth split)

cv_type = "expanding"  # "block" (default) or "expanding" or "randomized" (not recommended for time series and panels)
nfold = 4          # number of folds for cv (default 4). Irrelevant if cv_type = "expanding".

# for cv_type = "expanding" 
cv_first_date    = 197001   # start date for expanding window cv. Another example (Julia code): first_date = Date("2017-12-31", Dates.DateForma("y-m-d"))       
cv_block_periods = 120      # number of periods (months in this dataset): if cv_type="block", this is the block size

```
### Import and prepare data 

```r

library(data.table)

df = fread("HTBoost/examples/data/GlobalEquityReturns.csv") # import data as dataframe. Monthly LOG excess returns.

y = if (log_ret) 100 * df$excessret else 100 * (exp(df$excessret))
fnames = c("logCAPE", "momentum", "vol3m", "vol12m")    # will be an input (optional) to HTBdata().

# select features, and form a Julia DataFrame 
features_vector = c("logCAPE", "momentum", "vol3m", "vol12m")
x = DataFrames$DataFrame(df[, ..features_vector])     

# Translate the R dataframe to a Julia DataFrame
x = DataFrames$DataFrame(x)

```

### Set up models depending on selected option for cross-validation. Fit, print some output.

The panel does not need to be chronologically sorted.

```r

if (cv_type == "randomized") {
  param = HTBoost$HTBparam(nfold=nfold, overlap=overlap, loss=loss, modality=modality, priortype=priortype, randomizecv=TRUE)
} else if (cv_type == "block") {   # default 
  param = HTBoost$HTBparam(nfold=nfold, overlap=overlap, loss=loss, modality=modality, priortype=priortype)
} else if (cv_type == "expanding") {
  df_julia = DataFrames$DataFrame(df)   # dataframe including "dates" 
  indtrain_a = HTBoost$HTBindexes_from_dates(df_julia,"dates", cv_first_date, cv_block_periods)$indtrain_a
  indtest_a = HTBoost$HTBindexes_from_dates(df_julia,"dates", cv_first_date, cv_block_periods)$indtest_a
  param = HTBparam(nfold=nfold, overlap=overlap, loss=loss, modality=modality, priortype=priortype, indtrain_a=indtrain_a, indtest_a=indtest_a)
}

data = HTBoost$HTBdata(y,x,param,dates,fnames=fnames)
output = HTBoost$HTBfit(data, param)

yhat = HTBoost$HTBpredict(x, output)  # in-sample fitted value.

cat("\n depth =", output$bestvalue, ", number of trees =", output$ntrees, "\n")
cat(" in-sample R2 =", round(1.0 - sum((y - yhat)^2) / sum((y - mean(y))^2), digits=3), "\n")

```

### Feature importance and smoothness; partial dependence plot 

See [Basic_use](Basic_use.md)