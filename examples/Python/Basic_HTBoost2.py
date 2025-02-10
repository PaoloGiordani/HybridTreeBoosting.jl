#
# Using HTBoost in Python with juliacall package
#
# A limitation is that, unlike what happens in Julia, compile time is paid every time the code is run.
# There may be ways around this using a persitent Julia session with pyjulia. 
#  
# Install juliacall package in Python (pip install juliacall)
# Install Distributed, HTBoost packages in Julia. This can also be done from Python using juliacall, as
# jl.seval("using Pkg")
# jl.seval("Pkg.add('Distributed')")
# jl.seval("Pkg.add('HTBoost')")

# import packages 
from juliacall import Main as jl, convert as jlconvert
import numpy as np
import pandas as pd

# Define sample size 
n = 1000 
p = 5     # number of features (at least 2) . The first one will be categorical. 

# Create random data 
x = np.zeros((n,p-1))

for i in range(1,p-1):
    x[:,i] = np.random.normal(0,1,n)

# test 2: categorical data in the first column
x_cat = np.random.choice(['value1', 'value2', 'value3'], size=n)
Eyx =  np.where(x_cat == 'value1', 0, 1) + x[:,1]
y   = Eyx + np.random.normal(0,1,n)

# test 1: Insert missing values (NaNs)
x[10,0] = np.nan
x[20,0] = np.nan
x[30,0] = np.nan


# Build pandas dataframe from x_cat and x 
df = pd.DataFrame(x, columns=[f'x{i+1}' for i in range(1,p)])

df.insert(0, 'x1', x_cat)
summary = df.describe(include='all')
print("Dataframe summary:\n", summary)

# HTBoost does not work with pandas dataframes. A simple option is to convert to array using x = df.values. This will only work if all values are numerical.
# A better option is to use the Julia DataFrame type. This is done by x = jl.DataFrame(df). y can be an array or  This will work with mixed data types.

jl.seval("using DataFrames")  
x = jl.DataFrame(df)
jl.describe(x)   
# y can be an array. If it is a dataframe, use y = jl.DataFrame(df_y)

# set up HTBoost 
jl.seval("using HTBoost")   # HTBoost must be installed in Julia 

# Set desired number of workes (here 8)
# Python incurs this compile time cost every time the program is run (unlike Julia and R)
jl.seval("using Distributed")
jl.seval("number_workers = 8; nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)")
jl.seval("@everywhere using HTBoost")

param  = jl.HTBparam(nfold=1,depth=2,nofullsample=True,modality='fast')
data   = jl.HTBdata(y,x,param)

import time

start_time = time.time()
output = jl.HTBfit(data,param)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Execution time 1st run: {elapsed_time} seconds")

start_time = time.time()
output = jl.HTBfit(data,param)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Execution time 2nd run: {elapsed_time} seconds")


yhat   = jl.HTBpredict(x,output)

corr  = np.corrcoef(Eyx,yhat)[0,1]

print(" ntrees:\n", output.ntrees)
print(" correlation of fitted values and E(y|x):\n", corr)

tuple1 = jl.HTBweightedtau(output,data,verbose=True)

# output can be in the form of a named tuple
tuple2 = jl.HTBpartialplot(data,output,[1,2])  # first variable is [1], not [0]
q   = tuple2.q     
pdp = tuple2.pdp

# alternatively, output can be in the form of separate variables 
q,pdp = jl.HTBpartialplot(data,output,[1,2])  # first variable is [1], not [0]
