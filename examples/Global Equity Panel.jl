"""

Working with time series and longitudinal data (panels).

- User only needs to provide a vector of dates in SMARTdata() and, if the y is overlapping, the overlap parameter in SMARTparam().
  Example: 
  param  = SMARTparam(overlap=20)        
  data   = SMARTdata(y,x,param,dates,fnames = fnames)
  where y,x and dates can be dataframes, e.g. y = df[:,:excessret], x = df[:,features_vector], dates = df[:,:date]
- Overlap defaults to 0. Typically overlap = h-1, where y(t) = Y(t+h)-Y(t). Used for purged-CV and to calibrate loglikdivide.
- By default, SMARTboost uses block-cv (aka purged cv), which is suitable for time series and longitudinal data. 
  To use expanding window cross-validation instead, provide indtrain_a and indtest_a in SMARTparam():
  the function SMARTindexes_from_dates() assists in building these indexes.
  Example: 
  first_date = Date("2017-12-31", Dates.DateFormat("y-m-d"))
  indtrain_a,indtest_a = SMARTindexes_from_dates(df,:date,first_date,12)  # 12 periods in each block, starting from first_datae
 
See SMARTindexes_from_dates() for more details.   

paolo.giordani@bi.no
"""

# On multiple cores
number_workers  = 8  # desired number of workers
using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using SMARTboostPrivate

using Random, Plots, CSV, DataFrames, Statistics

Random.seed!(12345)

# USER'S INPUTS 

# data
log_ret        = true    # true for log returns, false for returns
overlap        = 0       # 0 for non-overlapping (default), h-1 for overlapping

# SMARTboost

loss       = :L2   # :L2, :Huber, :logistic, :quantile
modality   = :fast  
priortype  = :hybrid   #:hybrid

cv_type     = "expanding"  # "block" or "expanding" or "randomized" (not recommended for time series and panels)

# if cv_type = "expanding" 
cv_first_date     = 197001   # start date for expanding window cv       
cv_block_periods  = 120      # number of periods (months in this dataset): if cv_type="block", this is the block size

# END USER'S OPTIONS

df = CSV.read("examples/data/GlobalEquityReturns.csv", DataFrame, copycols = true) # import data as dataframe. Monthly LOG excess returns.
display(describe(df))

# prepare data 
features_vector = [:logCAPE, :momentum, :vol3m, :vol12m ]
log_ret ? y     = 100*df[:,:excessret] : y  = @. 100*(exp(df[:,:excessret]) - 1.0 )
x      = df[:,features_vector]
fnames = ["logCAPE", "momentum", "vol3m", "vol12m"  ]

# set up SMARTparam and SMARTdata, then fit, depending on cross-validation type

if cv_type == "randomized"
  param  = SMARTparam(overlap=0,loss=loss,modality=modality,priortype=priortype,randomizecv=true) 
elseif cv_type == "block"   # default 
  param  = SMARTparam(overlap=0,loss=loss,modality=modality,priortype=priortype) 
elseif cv_type == "expanding"
  indtrain_a,indtest_a = SMARTindexes_from_dates(df,:date,cv_first_date,cv_block_periods)
end 

data   = SMARTdata(y,x,param,df[:,:date],fnames = fnames)
output = SMARTfit(data,param)


# Extracting in-sample fitted value.
yfit      = SMARTpredict(x,output)

println("\n depth = $(output.bestvalue), number of trees = $(output.ntrees) ")
println(" in-sample R2 = ", round(1.0 - sum((y - yfit).^2)/sum((y .- mean(y)).^2),digits=3) )

# feature importance
fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output,data);

# partial dependence plots, best four features. q1st is the first quantile. e.g. 0.01 or 0.05
q,pdp  = SMARTpartialplot(data,output,sortedindx[[1,2,3,4]],q1st=0.01,npoints = 5000)

# partial dependence plots
pl = Vector(undef,4)

for i in 1:4 
  pl[i]   = plot(q[:,i],pdp[:,i], legend=false,title=fnames[sortedindx[i]],color=:green)
end 

display(plot(pl[1],pl[2],pl[3],pl[4], layout=(2,2), size=(1200,600)))  # display() will show it in Plots window.
