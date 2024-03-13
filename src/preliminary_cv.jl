#=
Runs preliminary cross-validation, and sets param which will then be used in all the subsequent models.
The best model from this exercise is not included in the stack.
To speed up the process, a fairly high lambda and low depth are used whenever reasonable. 


preliminary_cv!
preliminary_cv_categoricals!


1) For categorical data, if modality is not :fastest, runs a preliminary cv to select n0_cat and  mean_encoding_penalization.
   This is done with a double loop. 

   for n0 
     for mean_encoding_penalization 

   which is speeded up (broken) by assuming that when increasing n0, mean_encoding_penalization cannot increase
   (the assumption seems reasonable and is supported in my simulations). 


=#


# modifies param0.n0_cat and param0.mean_encoding_penalization
function preliminary_cv!(param0,data)   

    if param0.modality in [:fast,:fastest]
        return
    end 
    
    preliminary_cv_categoricals!(param0,data)

end     



function preliminary_cv_categoricals!(param0,data)

    if isempty(param0.cat_features)
        return
    end 

    param_cv = deepcopy(param0)

    param_cv.modality = :fast
    param_cv.lambda = 0.3 
    param_cv.depth  = 3
    param_cv.verbose = :Off 

    T = param0.T 

    n0_multiplier_a = T.([1 10 100])
    mep_a = T.([0.0, 0.25, 0.5])
   
    mep_a0 = copy(mep_a)

    loss0 = T(Inf)
    loss_a = fill(T(Inf),length(n0_multiplier_a),length(mep_a))

    for (i,n0) in enumerate(n0_multiplier_a)    # n0 here is a multiplier (not the actual n0, which cannot be 1 for, say, :quantile)

        for (j,mep) in enumerate(mep_a)

            param_cv.n0_cat = param0.n0_cat*n0
            param_cv.mean_encoding_penalization = mep
            output = HTBfit(data,param_cv) 
            loss_a[i,j] = output.loss

            if output.loss < loss0
                loss0 = output.loss
            else 
                 mep_a = mep_a[1:j-1]        # if mep_a = T.([0.0, 0.25, 0.5])
                break
            end

        end 

        length(mep_a)==1 ? break : nothing
     
    end

    minloss,cartesian_index = findmin(loss_a)
    param0.n0_cat = param0.n0_cat*n0_multiplier_a[cartesian_index[1]]
    param0.mean_encoding_penalization = mep_a0[cartesian_index[2]]

end 