
# Installation and Introduction to the R bindings of HTBoost.jl 

## Install Julia  

Ensure Julia is installed. 
Julia can be downloaded from (https://julialang.org/downloads/)

## Each script should start with the following 4 steps 
#### 1. Ensure that the Julia executable is in the system search path.

In most cases this should not be necessary, but if R cannot find the path:  

- To find the path to the Julia executable, run the following command in Julia:
  ```julia Sys.BINDIR ```

- Set the path at the start of your R script (only the path, excluding julia.exe)

```r
julia_path = "C:\\Users\\.julia\\juliaup\\julia-1.11.2+0.x64.w64.mingw32\\bin"  # replace with your path
Sys.setenv(JULIA_BINDIR = julia_path)
```

#### 2. Install JuliaConnectoR package (if not already installed) and load it

```r
if (!require(JuliaConnectoR)) {
  install.packages("JuliaConnectoR")
}

library(JuliaConnectoR)
```

#### 3. Install the Julia packages Distributed, DataFrames, HTBoost (if not already installed), and load them.

Installation is needed only once, and can be done from Julia or in R using the JuliaConnectoR package as follows:

```r
juliaEval('using Pkg; Pkg.add("Distributed")')
juliaEval('using Pkg; Pkg.add("DataFrames")')
```
To install HTBoost from the registry, use the following command.
```r
juliaEval('using Pkg; Pkg.add("HTBoost")')
```
Alternatively, this will work even if the package is not in the registry.
```r
juliaEval('using Pkg; Pkg.add("https://github.com/PaoloGiordani/HTBoost.jl")')
```


Load the packages 

```r
HTBoost = juliaImport("HTBoost")
DataFrames = juliaImport("DataFrames")   
#RData = juliaImport("RData")    # not required by HTBoost, can be convenient to work with R datasets in Julia
```

#### 4. Set the desired number of workers (cores) to be used in parallel.

```r
juliaEval('
           number_workers  = 8  # desired number of workers, e.g. 8
           using Distributed
           nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
           @everywhere using HTBoost
           ')
```


## Running HTBoost 

The most important functions in HTBoost are

```r
param  = HTBoost$HTBparam()           # defines the model     
data   = HTBoost$HTBparam(y,x,param)  # defines the data, including optional features such as names, weights, time
output = HTBoost$HTBfit(data,param)   # fits the model 
yf     = HTBoost$HTBpredict(x_predict,output)
t      = HTBoost$HTBweightedtau(output,data)  # variable importance and smoothness
t      = HTBoost$HTBpartialplot(data,output,c(1,2))  # partial dependence plots, here for the first two features  
```

PROVIDE LINK!!! Example and/or tutorial
See Basic_use.R for an example, and tutorials (LINK!). 
DISTINGUISH TUTORIALS IN R, WHERE NEEDED. AND TUTORIALS IN JULIA.

## Further notes on using HTBoost in R

### To translate the Julia tutorials in R 

- Apologies to all the R purists for often using = everywhere instead of distinguishing between functions and assignments (= and <-).
- My versioning of Julia code to R is probably amateurish. Feel free to improve it. 
- Change Julia symbols to R strings. e.g. :modality to "modality", :On to "On"
- Change Julia true/false to R TRUE/FALSE 
- Change Julia vectors [1,2,3] to R vectors c(1,2,3)
- The output of each function in Julia is a named tuple. This corresponds to a list in R, whose elements can be accessed in the usual way, e.g. 
```r 
  ntrees = output$ntrees  
```
  Should this fail, we can use juliaGet() to translate the Julia object into a proper R list, e.g.  
```r 
  list = juliaGet(output)
  ntrees = list$ntrees  
```
- In the tutorials the output of a function is sometimes given by several variables, e.g. (see Zero_inflated_y.md ADD LINK) in Julia 
```julia 
    yf,prob0,yf_not0     = HTBpredict(x_test,output)
```
This will not work in R, but it is always possible to work with the list, so R becomes
```r 
    t = HTBoost$HTBpredict(x_test,output)
    prob0 = t$prob0 
```    

### Getting your data from R to HTBoost

If y and x are numerical matrices, they can be taken into HTBdata() directly, e.g. 

```r
n      = 1000 
p      = 2  
x      = matrix(rnorm(n*p),nrow = n,ncol = p)
y      = x[,1] + rnorm(n)

data   = HTBoost$HTBparam(modality="accurate")  
data   = HTBoost$HTBdata(y,x,param)
output = HTBoost$HTBfit(data,param)
```

This will work even if there are missing data (*NA* in R, *NaN* or *missing* in Julia) (see [Missing data](tutorials/Missing.md)) for how HTBoost deals with missing data internally, delivering superior accuracy if the underlying function is at least partially smooth.  

When y and/or x contain strings (categorical features), we must translate our R dataframe into a Julia DataFrame, which is then fed to HTBdata(), e.g. (continuing from the previous example)

```r
x_string =  sample(c("v1", "v2", "v3"), n, replace = TRUE)   # create a categorical with 3 values
df       = data.frame(x,x_string)                          # R dataframe 
x        = DataFrames$DataFrame(df)                          # x is a Julia dataframe
data     = HTBoost$HTBdata(y,x,param,fnames=colnames(df))    # pass the column names 
output   = HTBoost$HTBfit(data,param)                        

```

Columns of string values are automatically interpreted by HTBoost as a categorical. If some categorical features are represented by numerical values, it is necessary to list them in param (in which case all categorical features, even strings, must be listed). This can be done either with a vector of their column positions, or with their names, if fnames (an optional argument) is provided to HTBdata()
```r
# either 
param = HTBoost$HTBparam(cat_features=c(3))
data  = HTBoost$HTBdata(y,x,param)    # passing the column names is optional

# or
param = HTBoost$HTBparam(cat_features=c("x_string"))
data  = HTBoost$HTBdata(y,x,param,fnames=colnames(df))    # passing the column names is required

```

