""" 

Modality = :accurate should cover the needs of most users. 

In situations where computing time is not an issue and even the smallest increment in performance matter, the following may be tried:

- Lower lambda to 0.05, particularly if the function is highly nonlinear (some features have high average τ values.)
- HTBboost cross-validates depth up to 6. This is sufficient in most circustances. If the best depth is 6, try 7 and perhaps even 8. (Note that computing time can easily double with each increment in depth.)
  This can be achieved by running, for example: 
  
  output = HTBfit(data,param,cv_grid=[5,6,7,8])

- Consider alternative loss functions, such as :L2loglink or :t.    

- If the signal-to-noise ratio is very low, set cv_sparsity = true. (The default sets it to true only if
  if n/p is not large, but very noisy data may benefit from it.)

  output = HTBfit(data,param,cv_sparsity=true)

""" 
