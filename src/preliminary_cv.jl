#=
Runs preliminary cross-validation, and sets param which will then be used in all the subsequent models.
The best model from this exercise is not included in the stack.
To speed up the process, a fairly high lambda and low depth are used whenever reasonable. 


preliminary_cv!
preliminary_cv_categoricals!


1) For categorical data, can run a preliminary cv to select n0_cat and  mean_encoding_penalization.
   What is cv depend on param.cv_categoricals, which can be :none,:n0,:penalty,:both
   This is done with a double loop. 

   for n0 
     for mean_encoding_penalization 

   which is speeded up (broken) by assuming that when increasing n0, mean_encoding_penalization cannot increase
   (the assumption seems reasonable and is supported in my simulations). 


=#


# modifies param.n0_cat and param.mean_encoding_penalization
function preliminary_cv!(param,data,indices)   
 
    preliminary_cv_categoricals!(param,data,indices)

end     


function preliminary_cv_categoricals!(param,data,indices)


    if isempty(param.cat_features)
        return
    end 

    if param.cv_categoricals==:none
        return
    end 

    # save values that will be modified in loop (to avoid deepcopy() and StackOverflow)
    lambda,verbose,depth = param.lambda,param.verbose,param.depth

    # values for fast cv 
    param.lambda = max(param.lambda,param.T(0.33)) 
    param.depth  = 4
    param.verbose = :Off

    T = param.T 
    n0_multiplier_a = T.([1])
    mep_a = T.([1])

    if param.cv_categoricals == :n0
        n0_multiplier_a = T.([0.1,1,10,100])
    elseif param.cv_categoricals == :penalty    
        mep_a = T.([0.0,0.5,1.0,2.0,4.0])
    elseif param.cv_categoricals == :both
        n0_multiplier_a = T.([0.1,1,10,100])
        mep_a = T.([0.0,0.5,1.0,2.0,4.0])
    else 
        @error "param.cv_categoricals must take values in [:none,:n0,:penalty,:both]"    
    end 

    mep_a0 = copy(mep_a)

    loss0 = T(Inf)
    loss_a = fill(T(Inf),length(n0_multiplier_a),length(mep_a))

    for (i,n0) in enumerate(n0_multiplier_a)    # n0 here is a multiplier (not the actual n0, which cannot be 1 for, say, :quantile)

        for (j,mep) in enumerate(mep_a)

            param.n0_cat = param.n0_cat*n0
            param.mean_encoding_penalization = mep
            ntrees,loss,meanloss,stdeloss,HTBtrees1st,indtest,gammafit_test,y_test,problems = HTBsequentialcv(data,param,indices=indices)
            loss_a[i,j] = loss 


            if loss < loss0
                loss0 = loss
            else 
                 mep_a = mep_a[1:j-1]        # if mep_a = T.([0.0, 0.25, 0.5])
                break
            end

        end 

        length(mep_a)==1 ? break : nothing
     
    end

    minloss,cartesian_index = findmin(loss_a)
    param.n0_cat = param.n0_cat*n0_multiplier_a[cartesian_index[1]]
    param.mean_encoding_penalization = mep_a0[cartesian_index[2]]

    # reinstate original values
    param.lambda = lambda 
    param.depth  = depth 
    param.verbose = verbose
    

end 