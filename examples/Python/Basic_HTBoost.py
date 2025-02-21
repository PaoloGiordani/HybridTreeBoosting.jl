#
# using HybridTreeBoosting in Python with juliacall package
#
# Simplest example with one feature.
#
# A limitation is that, unlike what happens in Julia, compile time is paid every time the code is run.
#
# The code seems slow .... 

# Install juliacall package in Python 
# Install Distributed, HTBoost packages in Julia. This can also be done from Python using juliacall, as
# jl.seval("using Pkg")
# jl.seval("Pkg.add('Distributed')")
# jl.seval("Pkg.add('HybridTreeBoosting')") 

# import packages 
from juliacall import Main as jl, convert as jlconvert
import numpy as np
import pandas as pd

# Define sample size 
n = 1000 

# Create random data 
x  = np.random.normal(0,1,n)
x  = np.sort(x)
u  = np.random.normal(0,1,n)

Eyx  = x 
y    = Eyx + u

# HTBoost does not work with pandas dataframes. A simple option is to convert to array using x = df.values. This will only work if all values are numerical.
# A better option is to use the Julia DataFrame type. This is done by x = jl.DataFrame(df). y can be an array or  This will work with mixed data types.

fnames = ['x1']
df = pd.DataFrame(x, columns=fnames)  # convert to pandas dataframe
jl.seval("using DataFrames")  
x = jl.DataFrame(df)
# y can be an array. If it is a dataframe, use y = jl.DataFrame(df_y)


# run HTBoost 

jl.seval("using HybridTreeBoosting")   # HTBoost must be installed in Julia 

# Set desired number of workes (here 4)
jl.seval("using Distributed")
jl.seval("number_workers = 4; nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)")
jl.seval("@everywhere using HybridTreeBoosting")

param  = jl.HTBparam(nfold=1,depth=2,nofullsample=True,modality="fastest")
data   = jl.HTBdata(y,x,param)

output = jl.HTBfit(data,param)
yhat   = jl.HTBpredict(x,output)

corr  = np.corrcoef(Eyx,yhat)[0,1]

print(" ntrees:\n", output.ntrees)
print(" correlation of fitted values and E(y|x):\n", corr)

tuple1 = jl.HTBweightedtau(output,data,verbose=True)
tuple2 = jl.HTBpartialplot(data,output,[1])  # first variable is [1], not [0]

q  = tuple2.q     # partial dependence plot, x-values  
pdp = tuple2.pdp  # partial dependence plot, y-values
