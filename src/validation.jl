#
# Functions for validation (early stopping) and CV
#
# indicescv
# indicescv_panel_purge  indices for purged cv. Extends De Prado's purged cv to panel data.
# SMARTsequentialcv      key function: fits nfold sequences of boosted trees, and selects number of trees by cv.
#                        Computationally efficient validation/cv of number of trees in boosting
#
# Note: SMARTindices_from_dates (for expanding window cv) is in main_functions.jl since it is called by user.


"""
    indicescv(j,nfold,sharevalidation,n,indices)
indtrain,indtest = indicescv(j,nfold,sharevalidation,n,indices)
Intended use: ytrain = data.y[indtrain], xtrain = data.x[indtrain,:], ytest = data.y[indtest], xtest = data.x[indtest,:]
indices = [i for i in 1:n] for contiguous blocks, indices = shuffle([i for i = 1:n]) for randomized train and test data
"""
function indicescv(j::I,nfold::I,sharevalidation,n::I,indices::Vector{I}) where I<:Int  # j = 1,...,nfold

    if nfold == I(1)
        typeof(sharevalidation)==I ? samplesize = sharevalidation : samplesize = I(floor(n*sharevalidation))
        starti     = n - samplesize + 1
        indtest    = indices[[i for i in starti:n]]
        indtrain   = indices[[i for i in 1:starti-1]]
    else
        samplesize = I(floor(n/nfold))
        starti     = I(1+(samplesize*(j-1)))
        j==nfold ? endi = n : endi   = samplesize*j
        indtest       = indices[[i for i in starti:endi]]

        if j==1
            indtrain = indices[[i for i in endi+1:n]]
        elseif j==nfold
            indtrain = indices[[i for i in 1:starti-1]]
        else
            indtrain = vcat(indices[[i for i in 1:starti-1]],indices[[i for i in endi+1:n]])
        end

    end

    return indtrain,indtest

end




"""
    indicescv_panel_purge(j,nfold,sharevalidation,n,indices,dates,param.overlap)

indtrain,indtest = indicescv_panel_purge(j,nfold,sharevalidation,n,indices,dates,param.overlap)

Extends to panel data the "purged cross-validation" of De Prado (which in turns is closely related to hv-Blocked CV of Racine) -- proposed for a single time series ---
to allow for both overlapping and/or panel data features (several observations sharing the same date.)
For a good academic reference, see Oliveira et al. (2021), "Evaluation Procedures for Forecasting with Spatiotemporal Data"

Sets the test sample with no concern for the date, and then purges from the training set observations with the same date. If overlap > 0, purges additional dates.
Not the most statistically efficient way to split the data, which would be to have test set start at the first observation of a new date (and end at the last), but
the loss of efficiency should be small. All data points are part of a test set (most important), but the training sets are not as large as they could be.

"""
function indicescv_panel_purge(j::I,nfold::I,sharevalidation,n::I,indices::Vector{I},dates::AbstractVector,overlap::I)  where I<:Int  # j = 1,...,nfold)

    indtrain,indtest  = indicescv(j,nfold,sharevalidation,n,indices)  # plain-vanilla split

    dateI   = dates[indtest[1]]
    dateF   = dates[indtest[end]]
    datesu  = unique(dates)

    if length(datesu)==length(dates) && overlap==I(0)
        return indtrain,indtest
    else                                       # purge from train set all indexes sharing the same date with test set, which is assumed ordered
        indtrain = hcat(indtrain,dates[indtrain])
        indtrain = indtrain[indtrain[:,2] .!= dateI,:]
        indtrain = indtrain[indtrain[:,2] .!= dateF,:]

        if overlap>I(0)                   # If there is overlap, purge more dates
            tI     = argmax(datesu .== dateI)  # index of dateI at datesu. The expression in parenthesis is only true for one value
            tF     = argmax(datesu .== dateF)  # index of dateF at datesu

            for o = 1:overlap

                if j>1
                    dateI   = datesu[tI-o]
                    indtrain = indtrain[indtrain[:,2] .!= dateI,:]
                end

                if j<nfold
                    dateF    = datesu[tF+o]
                    indtrain = indtrain[indtrain[:,2] .!= dateF,:]
                end

            end
        end

        return (indtrain[:,1],indtest)
    end

end



"""
    SMARTsequentialcv( data::SMARTdata, param::SMARTparam; .... )

ntrees,loss,meanloss,stdeloss,SMARTtrees = SMARTsequentialcv( data::SMARTdata, param::SMARTparam; .... )

sequential cross-validation for SMART. validation (early stopping) or n-fold cv for growing ensemble, automatically selecting the cv-optimal number of trees for a given parameter vector.

# Inputs (optional)

- The following options are specified in param::SMARTparam
     randomizecv,nfold, sharevalidation,stderulestop,losscv

# Output:

- ntrees                 Number of trees chosen by cv (or validation)
- loss                   Average loss in nfold test samples evaluated at ntrees
- meanloss               Vector of loss (mean across nfolds) at 1,2,...,J, where J>=ntrees
- stdeloss               Vector of stde(loss) at 1,2,...,J, where J>=ntrees; standard error of the estimated mean loss computed by aggregating loss[i] across i = 1,....,n, so std ( l .- mean(l)  )/sqrt(n).
                         Unlike the more common definition, this applies to p = 1 as well.
- SMARTtrees1st          ::SMARTboostTrees fitted on y_train,x_train for first fold (intended for nfold = 1)
- indtest                nfold vector of vectors of indices of validation sample
- gammafit_test          vector of fitted values for nfold test samples at lowest loss (corresponding to ntrees), to be compared with y_test
- ytest                  vector of test observations
- problems               true if there is a computational problem 
"""
function SMARTsequentialcv( data::SMARTdata, param::SMARTparam; indices=Vector(1:length(data.y)) )

    T = typeof(param.lambda)
    I = typeof(param.depth)

    nfold,sharevalidation,stderulestop,n = param.nfold, param.sharevalidation, param.stderulestop, I(length(data.y))

    data_a           = Array{SMARTdata}(undef,nfold)    
    rh_a             = Array{NamedTuple}(undef,nfold)
    gammafit_test_a  = Array{Vector{T}}(undef,nfold)     # used to compute cv loss 
    gammafit_test_ba_a = Array{Vector{T}}(undef,nfold)   # bias-adjusted: used to output predictions 
    t_a              = Array{Tuple}(undef,nfold)
    SMARTtrees_a     = Array{SMARTboostTrees}(undef,nfold)
    indtest_a        = Vector{Vector{I}}(undef,nfold)
    param_a          = fill(param,nfold)
    problems = 0

    for nf in 1:nfold

        if isempty(param.indtrain_a)    # no user-provided train-test allocation
            indtrain,indtest         = indicescv_panel_purge(nf,nfold,sharevalidation,n,indices,data.dates,param.overlap)
        else 
            indtrain,indtest         = param.indtrain_a[nf],param.indtest_a[nf]
        end
      
        data_nf = SMARTdata(data.y[indtrain],data.x[indtrain,:],data.weights[indtrain],data.dates[indtrain],data.fnames,param_a[nf].cat_features)
        param_nf,data_nf,meanx,stdx          = preparedataSMART(data_nf,param)
        τgrid,μgrid,info_x,n_train,p         = preparegridsSMART(data_nf,param_nf,meanx,stdx)
        gamma0                               = initialize_gamma(data_nf,param_nf)
        gammafit                             = fill(gamma0,length(indtrain))

        param_a[nf] = updatecoeff(param_nf,data_nf.y,gammafit,data_nf.weights,0)
        SMARTtrees_a[nf]    = SMARTboostTrees(param_a[nf],gamma0,n_train,p,meanx,stdx,info_x)
        rh_a[nf],param_a[nf]= gradient_hessian(data_nf.y,data_nf.weights,SMARTtrees_a[nf].gammafit,param_a[nf],0)
        gammafit_test_a[nf] = gamma0*ones(I,length(indtest))
        gammafit_test_ba_a[nf] = gamma0*ones(I,length(indtest))
        t_a[nf]             = (indtrain,indtest,meanx,stdx,n_train,p,τgrid,μgrid,info_x,param_a[nf])
        indtest_a[nf]       = indtest
        data_a[nf]          = data_nf

    end

    lossM,meanloss,stdeloss,j = zeros(T,param.ntrees,nfold),zeros(T,param.ntrees),zeros(T,param.ntrees), I(0)
    gammafit_test0 = T[]

    # Preliminay run to calibrate coefficients and priors 
    for nf in 1:nfold

        indtrain,indtest,meanx,stdx,n_train,p,τgrid,μgrid,info_x,param = t_a[nf]
        data_nf = data_a[nf]

        Gβ,trash = fit_one_tree(data.y[indtrain],data.weights[indtrain],SMARTtrees_a[nf],rh_a[nf].r,rh_a[nf].h,data_nf.x,μgrid,info_x,τgrid,param_a[nf])
        param_a[nf] = updatecoeff(param_a[nf],data.y[indtrain],SMARTtrees_a[nf].gammafit+Gβ,data.weights[indtrain],0) # +Gβ, NOT +λGβ
        trash,param_a[nf] = gradient_hessian( data.y[indtrain],data.weights[indtrain],SMARTtrees_a[nf].gammafit+Gβ,param_a[nf],1)            

    end
 
    for i in 1:param.ntrees

        gammafit_test = T[]
        lossv  = T[]

        for nf in 1:nfold

            indtrain,indtest,meanx,stdx,n_train,p,τgrid,μgrid,info_x,param = t_a[nf]
            data_nf = data_a[nf]
    
            x_test  = preparedataSMART_test(data.x[indtest,:],param_a[nf],meanx,stdx)
            Gβ,ij,μj,τj,mj,βj,fi2j = fit_one_tree(data.y[indtrain],data.weights[indtrain],
                    SMARTtrees_a[nf],rh_a[nf].r,rh_a[nf].h,data_nf.x,μgrid,info_x,τgrid,param_a[nf])

            param_a[nf] = updatecoeff(param_a[nf],data.y[indtrain],SMARTtrees_a[nf].gammafit+Gβ,data.weights[indtrain],i) # +Gβ, NOT +λGβ
            updateSMARTtrees!(SMARTtrees_a[nf],Gβ,SMARTtree(ij,μj,τj,mj,βj,fi2j),i,param_a[nf])
            rh_a[nf],param_a[nf] = gradient_hessian( data.y[indtrain],data.weights[indtrain],SMARTtrees_a[nf].gammafit,param_a[nf],2)
            gammafit_test_a[nf] = gammafit_test_a[nf] + param.lambda*SMARTtreebuild(x_test,ij,μj,τj,mj,βj,param_a[nf])
            bias,gammafit_test_ba_a[nf] = bias_correct(gammafit_test_a[nf],data.y[indtrain],SMARTtrees_a[nf].gammafit+Gβ,param)
 
            lossM[i,nf],losses  = losscv(param_a[nf],data.y[indtest],gammafit_test_a[nf],data.weights[indtest] )  # lossv is a (ntest) vector of losses. 
            lossv = vcat(lossv,losses)
            gammafit_test = vcat(gammafit_test,gammafit_test_ba_a[nf])  # bias-adjusted (used in stacking), while loss is computed on original fit

        end

        meanloss[i]  = mean(lossM[i,:])
        stdeloss[i]  = std( lossv .- meanloss[i],corrected = false )/sqrt(length(lossv))  # std from all observations

        if i==1
            gammafit_test0 = copy(gammafit_test)
        elseif meanloss[i]<meanloss[i-1]
            gammafit_test0 = copy(gammafit_test)
        end

        displayinfo(param.verbose,i,meanloss[i],stdeloss[i])

        # warn if loss jumps up by more than 5 standard errors
        if i>5 && param.warnings==:On
            if ((meanloss[i]-meanloss[i-1])/stdeloss[i-1])>5  
                @warn "Cross validation loss jumped at tree number $i. This typically signals a numerical problem, 
                and mayflag sub-par model performance in situations of near-perfect fit."
                problems = problems + 1
            end     
        end


        # break the loop if CV loss is either increasing or decreasing too slowly in last 10% of iterations
        J = maximum([20,I(floor(i/10))])
        J2 = I(floor(J/2))
        if i>=J
            sdiff = (mean(meanloss[i-J2+1:i]) - mean( meanloss[i-J+1:i-J2]))/(stdeloss[i]/sqrt(J2))                 
        else
            sdiff = T(-Inf)
        end

        if sdiff>-stderulestop || meanloss[i]==Inf
            break
        else
            j = j+1
        end

    end

    ntrees         = argmin(meanloss[1:j])    
    loss           = meanloss[ntrees]

    y_test = T[]

    for nf in 1:nfold
        y_test = vcat(y_test,data.y[indtest_a[nf]])
    end

    if ntrees==param.ntrees && param.warnings==:On
        @warn "The maximum number of trees $(param.ntrees) has been reached with CV loss still decreasing."
        problems = problems +1
    elseif ntrees==1 && param.warnings==:On
        @warn "Cross validation selects one tree. Reducing the learning rate lambda is recommended."
        problems = problems +1
    end

    if nfold==1
        SMARTtrees_a[1].trees = SMARTtrees_a[1].trees[1:ntrees]
       # bias adjustment 
        indtrain,indtest,meanx,stdx,n_train,p,τgrid,μgrid,info_x = t_a[1]
        bias,SMARTtrees_a[1].gammafit = bias_correct(SMARTtrees_a[1].gammafit,data.y[indtrain],SMARTtrees_a[1].gammafit,param)
        SMARTtrees_a[1].gamma0 +=  bias
    end

    return ntrees,loss,meanloss[1:j],stdeloss[1:j],SMARTtrees_a[1],indtest_a,gammafit_test0,y_test,problems

end
