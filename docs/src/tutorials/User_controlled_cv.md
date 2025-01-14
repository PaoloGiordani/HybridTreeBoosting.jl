## User's controlled cross-validation of parameters

**Summary**

- HTBfit() already parsimoniously cv the most important hyperparameters if modality in [:compromise,:accurate], and stacks the results. This should be sufficient for the majority of applications. 
- HTBcv() can be used to select additional (i.e. in addition to the internal cv in HTBfit) hyperparameters to cv. This should improve (or at least not hurt) accuracy in most cases but could be quite expensive.
- HTBcv() can also be used to replace HTBfit() and fully control the hyperparameter selection. (Not recommended.) 

**Discussion**

The recommended process for HTBoost is to use *modality* (:fast, :compromise, :accurate) in HTBfit() rather than HTBcv().
The various modalities in HTBfit() internally control the most important hyperparameters, being as parsimonious as possible due 
to the high computational costs of HTB. HTBfit() also stacks the models, which is not done here.
All modalities use early stopping to determine the number of trees, so ntrees should not be cv. 

The function HTBcv() is provided for users who want to cv additional parameters (not included in HTBfit) and can incur the computational costs. An example is provided below.

The function HTBcv() is also provided for advanced users who want to fully control the cross-validation process.
In most cases HTBcv() will be less efficient than HTBfit(), and probably less accurate. 
    
**Example of use: cv varlntau (strength of smoothness prior) in addition to the internal cv**

Set up model and data, including aspects of cv such as the number of folds and whether block-cv or randomized.

```julia 

param  = HTBparam(loss=:L2,modality=:accurate,nfold=1,randomize=false)  
data   = HTBdata(x,y,param)

```

Specify hyperparameters and values for cv as Array{Dictionary}. Here we cv only varlntau (strength of smoothness prior).
The resuls is a one-dimensional array. 

```julia 

params_cv = [Dict(
    :varlntau => varlntau)
    for
    varlntau in (0.25^2,0.5^1,1.0)
    ]

htbcv = HTBcv(data,param,params_cv)      # cv over dictionary AND internally
    
# Some info about the best model 
bestindex = htbcv.bestindex   # params_cv[bestindex] is for best set of cv hyperparameters       
bestparam = htbcv.bestparam   

# predict using best model
yf    = HTBpredict(x_oos,htbcv.output)    

```

**Example of use: fully control cv (not recommended for most users)**

Specify Array{Dictionary} of hyperparameters to be cv.

```julia 

params_cv = [Dict(
    :depth => depth,
    :varlntau => varlntau)    
    for
    depth in (2,4,6),
    varlntau in (0.25^2,0.5^1,1.0^2)
    ]
```

Call htbc with internal_cv = false

```julia

htbcv = HTBcv(data,param,params_cv,internal_cv=false)       

# Some info about the best model 
bestindex = htbcv.bestindex   # params_cv[bestindex] is for best set of cv hyperparameters       
bestparam = htbcv.bestparam   
output    = htbcv.output

# predict using best model
yf    = HTBpredict(x_oos,htbcv.output)    

```

