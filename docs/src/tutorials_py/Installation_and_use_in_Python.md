
# Installation and Introduction to the Python bindings of HTBoost.jl 

## Install Julia  

Install Julia, if required. This may not be necessary, as juliacall should automatically download a suitable version of Julia if required. 
Julia can be downloaded from (https://julialang.org/downloads/)

## Each script should start with the following 3 steps 

#### 1. Install juliacall package (*pip install juliacall*), then load it as

```py
from juliacall import Main as jl, convert as jlconvert
```

#### 2. Install the Julia packages Distributed, DataFrames, HTBoost, and load them.

Installation is needed only once, and can be done from Julia or in Python as follows:

```py
jl.seval("using Pkg")
jl.seval("Pkg.add('Distributed')")
jl.seval("Pkg.add('HybridTreeBoosting')")
```

To install HTBoost from the registry, use the following command.
```py
jl.seval('using Pkg; Pkg.add("HybridTreeBoosting")')
```
Alternatively, this will work even if the package is not in the registry.
```py
jl.seval('using Pkg; Pkg.add("https://github.com/PaoloGiordani/HybridTreeBoosting.jl")')
```

Load the packages in Julia

```py
jl.seval("using DataFrames")  
jl.seval("using HybridTreeBoosting")   # HTBoost must be installed in Julia 
```

#### 3. Set the desired number of workers (cores) to be used in parallel.

Note: Python incurs this compile time cost every time the program is run (unlike Julia and R). The compile cost increases in the number of workes, and is around 80'' for 8 workers.

```py
jl.seval("using Distributed")
jl.seval("number_workers = 8; nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)")
jl.seval("@everywhere using HybridTreeBoosting")
```

## Running HTBoost 

The most important functions in HTBoost are

```py
param  = jl.HTBparam()           # defines the model     
data   = jl.HTBparam(y,x,param)  # defines the data, including optional features such as names, weights, time
output = jl.HTBfit(data,param)   # fits the model 
yf     = jl.HTBpredict(x_predict,output)
t      = jl.HTBweightedtau(output,data)  # variable importance and smoothness
t      = jl.HTBpartialplot(data,output,[1,2])  # partial dependence plots, here for the first two features  
```

## Further notes on using HybridTreeBoosting in Python

### To translate the Julia tutorials in Python 

- Change Julia symbols to Python strings. e.g. :modality to 'modality' or "modality", :On to 'On' on "On"
- Change Julia true/false to True/False 
- The output of each function in Julia is a named tuple. Its elements can be accessed in the usual way, e.g. 
```py 
  ntrees = output.ntrees  
```

### Getting your data from Python to HTBoost

If y and x are numerical matrices with no missing data, they can be taken into HTBdata() directly, e.g. 

```py
x  = np.random.normal(0,1,n)
u  = np.random.normal(0,1,n)
y  = x + np.random.normal(0,1,n)

param   = jl.HTBparam(modality='accurate')  
data   = jl.HTBdata(y,x,param)
output = jl.HTBfit(data,param)
```

This will not work properly if there are missing data (*NaN* in Python, *NaN* or *missing* in Julia). See [Missing data](../tutorials/Missing.md)) for how HTBoost deals with missing data internally, delivering superior accuracy if the underlying function is at least partially smooth.  

When there are missing data, or when y and/or x contain strings (categorical features), we must work translate our Python dataframe into a Julia DataFrame, which is then fed to HTBdata(), e.g. (continuing from the previous example)

```py
fnames = ['x1']
df = pd.DataFrame(x, columns=fnames)  # convert to pandas dataframe
jl.seval("using DataFrames")  
x = jl.DataFrame(df)

data     = jl.HTBdata(y,x,param,fnames=colnames(df))    # pass the column names 
output   = jl.HTBfit(data,param)                        

```

Columns of string values are automatically interpreted by HTBoost as a categorical. If some categorical features are represented by numerical values, it is necessary to list them in param (in which case all categorical features, even strings, must be listed). This can be done with a vector of their column positions.

```py
param = jl.HTBparam(cat_features=[3])
data  = jl.HTBdata(y,x,param)    # passing the column names is optional

```

