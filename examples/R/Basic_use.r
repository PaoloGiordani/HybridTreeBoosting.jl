
# PG: this file is ready and runs fine. Can be used as a reference for the other files:
#    1) Add the information ad the head of Basic_example.jl 
#    2) Decide (later down) if HTBoost is in the registry or not. (minor point)
# *************************************************************************************************************

# Install Julia. 
# Ensure that the Julia executable is in the system search path.
# Provide the path to the directory.
# (In most cases this should not be necessary.) .
# To find the path to the Julia executable, run the following command in Julia:
# julia> Sys.BINDIR

julia_path = "C:\\Users\\A1810185\\.julia\\juliaup\\julia-1.11.2+0.x64.w64.mingw32\\bin"  # REPLACE with your path
Sys.setenv(JULIA_BINDIR = julia_path)

# Install JuliaConnectoR package (if not already installed) and load it
if (!require(JuliaConnectoR)) {
  install.packages("JuliaConnectoR")
}

library(JuliaConnectoR)

# Install the Julia packages Distributed, DataFrames, HTBoost (if not already installed).
# This can be done from Julia or in R using the JuliaConnectoR package as follows:
# juliaEval('using Pkg; Pkg.add("Distributed")')
# juliaEval('using Pkg; Pkg.add("DataFrames")')
# juliaEval('using Pkg; Pkg.add("RData")') # the RData package is not needed in this script (simulated data), but is generally useful for data exchange between R and Julia.

# To install HTBoost from the registry, use the following command.
# juliaEval('using Pkg; Pkg.add("HTBoost")')
# Alternatively, this will work even if the package is not in the registry.
# juliaEval('using Pkg; Pkg.add("https://github.com/PaoloGiordani/HTBoost.jl")')

# Load the Julia packages into R, if not already loaded. We only want to do this once to avoid recompliation time on each run.
# (This is not a problem in Julia, but it is when running Julia in R or Python). 
HTBoost = juliaImport("HTBoost") 
DataFrames = juliaImport("DataFrames") 
RData = juliaImport("RData")   # the RData package is not needed here (simulated data),  but is generally useful for data exchange between R and Julia.

# Set the desired number of workers (cores) to be used in parallel.
juliaEval('
           number_workers  = 1  # desired number of workers, e.g. 8
           using Distributed
           nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
           @everywhere using HTBoost
           ')

# Generate data for the example
n      = 1000
p      = 6
stde   = 1.0
n_test = 100000

f_1 = function(x,b)  b*x + 1 
f_2 = function(x,b)  2*sin(2.5*b*x)  
f_3 = function(x,b)  b*x^3
f_4 = function(x,b)  b*(x < 0.5) 
f_5 = function(x,b)  b/(1.0 + (exp(4.0*x )))
f_6 = function(x, b) {b * (x > -0.25 & x < 0.25)}

b1  = 1.5
b2  = 2.0
b3  = 0.5
b4  = 4.0
b5  = 5.0
b6  = 5.0

x      = matrix(rnorm(n*p),nrow = n,ncol = p)
x_test = matrix(rnorm(n_test*p),nrow = n_test,ncol = p)
f      = f_1(x[,1],b1) + f_2(x[,2],b2) + f_3(x[,3],b3) + f_4(x[,4],b4) + f_5(x[,5],b5) + f_6(x[,6],b6)
f_test = f_1(x_test[,1],b1) + f_2(x_test[,2],b2) + f_3(x_test[,3],b3) + f_4(x_test[,4],b4) + f_5(x_test[,5],b5) + f_6(x_test[,6],b6)
y      = f + rnorm(n)*stde

fnames = c("x1", "x2", "x3", "x4", "x5", "x6")

# When y and x are numerical matrices, we could feed them to HTBoost directly.
# However, a more general procedue (which allows strings), is to transform the data into a dataframe and then into a Julia DataFrame.
# which is done as follows (here only for x, but the same applies to y if it is not numerical).

df     =  data.frame(x)   # create a R dataframe
df_test = data.frame(x_test)
fnames = colnames(df)

x = DataFrames$DataFrame(df)
x_test = DataFrames$DataFrame(df_test)
DataFrames$describe(x)

# Set some HTBoost parameters
loss      = "L2"        # :L2 is default. Other options for regression are :L2loglink (if y≥0), :t, :Huber
modality  = "fast"      # "accurate", "compromise" (default), "fast", "fastest". The first two perform parameter tuning internally by cv.
priortype = "hybrid"    # "hybrid" (default) or "smooth" to force smoothness 
nfold     = 1           # number of cv folds. 1 faster (single validation sets), default 4 is slower, but more accurate.
nofullsample = TRUE     # if nfold=1 and nofullsample=TRUE, the model is not re-fitted on the full sample after validation of the number of trees
randomizecv = FALSE     # FALSE (default) to use block-cv. 
verbose     = "Off"

param = HTBoost$HTBparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,verbose=verbose,modality=modality,nofullsample=nofullsample)
data  = HTBoost$HTBdata(y,x,param,fnames=fnames)   # fnames is optional

# Fit the model
# The first run pays a heavy cost for compilation: around 1.5' for 8 workers. While in Julia this cost is paid only once,
# in R it is paid every time the program is run. To avoid this problem, run 

#output = HTBoost$HTBfit(data, param)

time_taken <- system.time({
  output <- HTBoost$HTBfit(data, param)
})

cat("HTBoost fitting time 1st run:", time_taken["elapsed"], "seconds\n")

time_taken <- system.time({
  output <- HTBoost$HTBfit(data, param)
})
cat("HTBoost fitting time 2nd run:", time_taken["elapsed"], "seconds\n")

# save (load) fitted model. NOTE: use the same filename as the name of the object that is being saved.
#save(output, file = "output")
#load("output")

# Fitted and Predicted values 
yhat = HTBoost$HTBpredict(x, output)      # Fitted values
yf = HTBoost$HTBpredict(x_test, output)   # Predicted values

# Print some information 
cat("modality =", param$modality, ", nfold =", nfold, "\n")
cat("depth =", output$bestvalue, ", number of trees =", output$ntrees, "\n")
cat("out-of-sample RMSE from truth", sqrt(sum((yf - f_test)^2)/n_test), "\n")

# Print information on importance and smoothness for each feature

# The Julia functions don't always print nearly in R. To control printing, create a R dataframe.
# 
tuple = HTBoost$HTBweightedtau(output,data,verbose=FALSE)  # output is a Julia tuple, which can be converted to a R list
list = juliaGet(tuple)

# Create a data frame: features from 1 to p 
df <- data.frame(
  feature = list$fnames,
  importance = list$fi,
  avgtau = list$avgtau_a
)

# Create a data frame: features sorted by importance
df_sorted <- data.frame(
  sortedindx = list$sortedindx,
  sorted_feature = list$fnames_sorted,
  sorted_importance = list$fi_sorted,
  sorted_avgtau = list$avgtau_a[list$sortedindx]
)

#  print("\n Variable importance and smoothness, from first to last feature (not sorted)")
  print(df)
print("\n Variable importance and smoothness, from most to least important feature")
print(df_sorted)

cat("\n Average smoothing parameter τ is", round(list$gavgtau, digits = 1), ".\n")
cat("\n In sufficiently large samples, and if modality is 'compromise' or 'accurate':\n")
cat(" - Values above 20-25 suggest very little smoothness in important features. HTBoost's performance may slightly outperform or slightly underperform other gradient boosting machines.\n")
cat(" - At 10-15 or lower, HTBoost should outperform other gradient boosting machines, or at least be worth including in an ensemble.\n")
cat(" - At 5-7 or lower, HTBoost should strongly outperform other gradient boosting machines.\n")


# Create a plot to visualize the average smoothness of the splits
library(ggplot2)
library(gridExtra)

df <- data.frame(x = list$x_plot, g = list$g_plot)

p <- ggplot(df, aes(x = list$x_plot, y = list$g_plot)) +
  geom_line() +
  labs(title = "avg smoothness of splits", x = "standardized x", y = "") +
  theme_minimal() + 
  theme(
    plot.title = element_text(size = 30, face = "bold"),
    axis.title.x = element_text(size = 20),
    axis.title.y = element_text(size = 20),
    axis.text = element_text(size = 20)
  )

print(p)

# Partial dependence plots 
tuple = HTBoost$HTBpartialplot(data,output,c(1,2,3,4,5,6))
list = juliaGet(tuple)   #  this is not necessary in most cases (e.g. tuple$q works fine)

q  = list$q     
pdp = list$pdp

library(ggplot2)
library(gridExtra)

# Assuming q and pdp are matrices and f_1, f_2, ..., f_6 are functions, b1, b2, ..., b6 are parameters
f_list <- list(f_1, f_2, f_3, f_4, f_5, f_6)
b_list <- list(b1, b2, b3, b4, b5, b6)

# Create an empty list to store the plots
pl <- vector("list", 6)

# Generate each plot
for (i in 1:length(pl)) {
  df <- data.frame(
    x = q[, i],
    HTB = pdp[, i],
    true = f_list[[i]](q[, i], b_list[[i]]) - f_list[[i]](q[, i] * 0, b_list[[i]])
  )
  
  pl[[i]] <- ggplot(df, aes(x = x)) +
    geom_line(aes(y = HTB, color = "HTB"), size = 1.5) +
    geom_line(aes(y = true, color = "true"), linetype = "dotted", size = 1) +
    labs(title = paste("PDP feature", i), x = "x", y = "f(x)") +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 15),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 12)
    ) +
    scale_color_manual(values = c("HTB" = "blue", "true" = "red"))
}

# Arrange the plots in a 3x2 grid layout
grid.arrange(grobs = pl, ncol = 3, nrow = 2)