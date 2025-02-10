# Install juliacall package in Python (pip install juliacall)
# Install Distributed, HTBoost packages in Julia. This can also be done from Python using juliacall, as
# jl.seval("using Pkg")
# jl.seval("Pkg.add('Distributed')")
# jl.seval("Pkg.add('HTBoost')")

# import packages 
from juliacall import Main as jl, convert as jlconvert
import numpy as np
import pandas as pd

jl.seval("using DataFrames")  
jl.seval("using HTBoost")   # HTBoost must be installed in Julia 

jl.seval("using Distributed")
jl.seval("number_workers = 8; nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)")
jl.seval("@everywhere using HTBoost")


# Generate data 

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


# Create a pandas dataframe

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

# HTBoost

loss      = "L2"        # :L2 is default. Other options for regression are :L2loglink (if y≥0), :t, :Huber
modality  = "fast"      # "accurate", "compromise" (default), "fast", "fastest". The first two perform parameter tuning internally by cv.
priortype = "hybrid"    # "hybrid" (default) or "smooth" to force smoothness 
nfold     = 1           # number of cv folds. 1 faster (single validation sets), default 4 is slower, but more accurate.
nofullsample = True     # if nfold=1 and nofullsample=TRUE, the model is not re-fitted on the full sample after validation of the number of trees
randomizecv = False     # FALSE (default) to use block-cv. 
verbose     = "Off"

randomizecv = False       # false (default) to use block-cv. 

param = jl.HTBparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,verbose=verbose,modality=modality,nofullsample=nofullsample)

data   = jl.HTBdata(y,x,param)   # fnames is optional

output = jl.HTBfit(data,param)

yhat = jl.HTBpredict(x, output)      # Fitted values
yf   = jl.HTBpredict(x_test, output)   # Predicted values

# Print information
print(f"out-of-sample RMSE from truth = {np.sqrt(np.sum((yf - f_test)**2) / n_test)}")

# importance and partial dependence
tuple = jl.HTBweightedtau(output,data,verbose=True)  # output is a named tuple which 

fnames,fi,fnames_sorted,fi_sorted,sortedindx = jl.HTBrelevance(output,data,verbose=False);
q,pdp  = jl.HTBpartialplot(data,output,[1,2,3,4,5,6]) # partial effects for the first 6 