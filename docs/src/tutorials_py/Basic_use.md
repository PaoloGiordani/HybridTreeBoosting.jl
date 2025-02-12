## Basic use

**Summary**

- Illustrates use of the main functions of HTBoost on a regression problem with simulated data. 
- param.modality as the most important user's choice, depending on time budget. 
- In default modality, HTBoost performs automatic hyperparameter tuning.
- The Python bindings use juliacall. Please read [Installation in Python](Installation_and_use_in_Python.md).
- Only a few tutorials are provided for Python. For more tutorials and examples, see [Julia tutorials](../Tutorials.md) and [Julia examples](../Examples.md), using the guidelines in [Installation in Python](Installation_and_use_in_Python.md) to adapt the code to Python. 


**Main points** 

- default loss is "L2". Other options for continuous y are "Huber", "t" (recommended in place of "Huber"), "gamma", "gammaPoisson",
  "L2loglink". For zero-inflated continuous y, options are "hurdleGamma", "hurdleL2loglink", "hurdleL2"   
- default is block cross-validation with nfolds=4: use randomizecv = TRUE to scramble the data. See [Global Equity Panel](../examples/Global_Equity_Panel.md) for further options on cross-validation (e.g. sequential cv, or generally controlling the training and validation sets).
 
- fit, with automatic hyperparameter tuning if modality is :compromise or :accurate
- save fitted model (upload fitted model)
- average τ (smoothness parameter), which is also plotted. (Smoother functions ==> larger gains compared to other GBM)
- feature importance
- partial effects plots


---
---
### Install and load required packages.

Follow steps 1-3 in [Installation in Python](Installation_and_use_in_Python.md).

### Generate data. 

y is the sum of six additive nonlinear functions, plus Gaussian noise.

```py
import numpy as np

# Define sample size
n = 10000
p = 6
stde = 1.0
n_test = 100000

# Define functions
def f_1(x, b):
    return b * x + 1

def f_2(x, b):
    return 2 * np.sin(2.5 * b * x)

def f_3(x, b):
    return b * x**3

def f_4(x, b):
    return b * (x < 0.5)

def f_5(x, b):
    return b / (1.0 + np.exp(4.0 * x))

def f_6(x, b):
    return b * ((x > -0.25) & (x < 0.25))

# Define coefficients
b1 = 1.5
b2 = 2.0
b3 = 0.5
b4 = 4.0
b5 = 5.0
b6 = 5.0

# Generate random data
x = np.random.normal(0, 1, (n, p))
x_test = np.random.normal(0, 1, (n_test, p))

# Compute f and f_test
f = f_1(x[:, 0], b1) + f_2(x[:, 1], b2) + f_3(x[:, 2], b3) + f_4(x[:, 3], b4) + f_5(x[:, 4], b5) + f_6(x[:, 5], b6)
f_test = f_1(x_test[:, 0], b1) + f_2(x_test[:, 1], b2) + f_3(x_test[:, 2], b3) + f_4(x_test[:, 3], b4) + f_5(x_test[:, 4], b5) + f_6(x_test[:, 5], b6)

# Generate y with noise
y = f + np.random.normal(0, stde, n)

# Define feature names
fnames = ["x1", "x2", "x3", "x4", "x5", "x6"]

```

When y and x are numerical matrices with no missing values, we could feed them to HTBoost directly.
However, a more general procedue (which allows strings and missing values), is to transform the data into a dataframe and then into a Julia DataFrame. This is done as follows (here only for x, but the same applies to y if it is not numerical).

```py

import pandas as pd

# Create a Pandas dataframe
df = pd.DataFrame(x)
df_test = pd.DataFrame(x_test)

# Define feature names
fnames = df.columns.tolist()

# Convert Pandas dataframe to Julia DataFrame
jl.seval("using DataFrames")
x = jl.DataFrame(df)
x_test = jl.DataFrame(df_test)

# Describe the Julia DataFrame
jl.seval("describe")(x)
```

**Options for HTBparam( ).**  

I prefer to specify parameter settings separately (here at the top of the script) rather than directly in HTBparam( ), which is of course also possible.  
modality is the key parameter: automatic hyperparameter tuning if modality is "compromise" or "accurate", no tuning (except of #trees) if "fast" or "fastest".    
In HTBoost, it is not recommended that the user performs 
hyperparameter tuning by cross-validation, because this process is done automatically if modality is
"compromise" or "accurate". The recommended process is to first run in modality="fast" or "fastest",
for exploratory analysis and to gauge computing time, and then switch to "compromise" (default)
or "accurate". For a tutorial on user-controlled cross-validation, see [User's controlled cross-validation.](../tutorials/User_controlled_cv.md)

```py

loss      = "L2"        # :L2 is default. Other options for regression are :L2loglink (if y≥0), :t, :Huber
modality  = "fast"      # "accurate", "compromise" (default), "fast", "fastest". The first two perform parameter tuning internally by cv.
priortype = "hybrid"    # "hybrid" (default) or "smooth" to force smoothness 
nfold     = 1           # number of cv folds. 1 faster (single validation sets), default 4 is slower, but more accurate.
nofullsample = True     # if nfold=1 and nofullsample=TRUE, the model is not re-fitted on the full sample after validation of the number of trees
randomizecv = False     # FALSE (default) to use block-cv. 
verbose     = "Off"
```

**Options for cross-validation:**

While the default in other GBM is to randomize the allocation to train and validation sets,
the default in HTBoost is block cv, which is suitable for time series and panels.
Set randomizecv=True to bypass this default. 
See [Time_series_and_panels](../tutorials/Time_series_and_panels.md) for further options on cross-validation (e.g. sequential cv, or generally controlling the training and validation sets).

```py
randomizecv = False       # False (default) to use block-cv. 
```

### Set up HTBparam and HTBdata, then fit. Optionally, save the model (or load it). Predict. Print some information. 

```py

param = jl.HTBparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,verbose=verbose,modality=modality,nofullsample=nofullsample)

data   = jl.HTBdata(y,x,param)   
output = jl.HTBfit(data,param)

yhat = jl.HTBpredict(x, output)      # Fitted values
yf   = jl.HTBpredict(x_test, output)   # Predicted values

# Print information
print(f"out-of-sample RMSE from truth = {np.sqrt(np.sum((yf - f_test)**2) / n_test)")

```

which prints

```py
out-of-sample RMSE from truth 0.3563147

```

### Feature importance and average smoothing parameter for each feature.  

```py
tuple = jl.HTBweightedtau(output,data,verbose=True)  # output is a named tuple which 
```

```markdown
 Row │ feature  importance  avgtau     sorted_feature  sorted_importance  sorted_avgtau 
     │ String   Float32     Float64    String          Float32            Float64
─────┼──────────────────────────────────────────────────────────────────────────────────
   1 │ x1          14.5666   0.458996  x3                        19.0633       3.30638
   2 │ x2          12.9643  19.6719    x5                        18.6942       3.72146
   3 │ x3          19.0633   3.30638   x6                        17.9862      35.1852
   4 │ x4          16.7254  36.0846    x4                        16.7254      36.0846
   5 │ x5          18.6942   3.72146   x1                        14.5666       0.458996
   6 │ x6          17.9862  35.1852    x2                        12.9643      19.6719

 Average smoothing parameter τ is 7.3.

 In sufficiently large samples, and if modality=:compromise or :accurate

 - Values above 20-25 suggest very little smoothness in important features. HTBoost's performance may slightly outperform or slightly underperform other gradient boosting machines.
 - At 10-15 or lower, HTBoost should outperform other gradient boosting machines, or at least be worth including in an ensemble.
 - At 5-7 or lower, HTBoost should strongly outperform other gradient boosting machines.
```

In this case smoothness is a mix of very different values across features: approximate linearity for x1, smooth functions for x3 and x5, and essentially sharp splits for x2, x4, and x6.
Note: Variable (feature) importance is computed as in Hastie et al., "The Elements of Statistical Learning", second edition, except that the normalization is for sum=100.  


Some examples of smoothness corresponding to a few values of tau (for a single split) help to interpret values of avgtau

<img src="../assets/Sigmoids.png" width="600" height="400">

On simulated data, we can evaluate the RMSE from the True f(x), exluding noise:


**Hybrid trees outperform both smooth and standard trees**

Here is the output for n=10k (nfold=1, nofullsample=True). 
Hybrid trees strongly outperform both smooth trees and standard symmetric (aka oblivious) trees. (Note: modality = :sharp is a very inefficient way to run a symmetric tree; use CatBoost or EvoTrees instead!)

```markdown
  modality = fastest, nfold = 1, priortype = hybrid
 depth = 5, number of trees = 141, gavgtau 7.3
 out-of-sample RMSE from truth 0.3136

modality = fastest, nfold = 1, priortype = smooth 
 depth = 5, number of trees = 121, gavgtau 4.5
 out-of-sample RMSE from truth 0.5751

 modality = fastest, priortype = sharp
depth = 5, number of trees = 183, avgtau 40.0
 out-of-sample RMSE from truth 0.5320
```

**Partial dependence plots**

Partial dependence assumes (in default) that other features are kept at their mean.

```py
fnames,fi,fnames_sorted,fi_sorted,sortedindx = jl.HTBrelevance(output,data,verbose=False);
q,pdp  = HTBpartialplot(data,output,[1,2,3,4,5,6]) # partial effects for the first 6 variables 

```

# plot partial dependence
```py
q,pdp = jl.HTBpartialplot(data,output,[1,2,3,4,5,6])
```

Here q and pdp are (npoints,6) matrices. Plotting each column (each feature) for different values of n. 
Partial plots for n = 1k,10k,100k, with modality = :fastest and nfold = 1.   
Notice how plots are smooth only for some features. 

### n = 1_000
<img src="../assets/Minimal1k.png" width="600" height="400">

### n = 10_000
<img src="../assets/Minimal10k.png" width="600" height="400">

### n = 100_000
<img src="../assets/Minimal100k.png" width="600" height="400">

