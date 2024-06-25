# Piecewise Linear Fit

### Find optimal number of break points for a piecewise linear fit

Methodology

	1. Run a simple linear regression (i.e. case with 0 break points)
	2. Set a min and max number of possible break points (eg. 1 to 4)
	3. Loop over possible breaks creating piecewise linear fit model for each
	4. Calculate RSS between actual and predicted values
	5. For each number of breaks, calculate Bayesian Information Criterion (BIC) value. 
	6. Identify optimal breaks as one which minimises BIC. BIC balances complexity with success of model fitting
	7. Enhance chosen model with ElasticNetCV for regularisation (mainly to prevent overfitting)
	

Possible use cases include:

  1. Outlier Detection  -->  Identifies outliers based on a specified threshold of standard deviations from the mean residual
  2. Extrapolation
 
 

![image](https://github.com/nweerasuriya/Python_Upskilling/assets/65176466/31d6e2c8-e005-4f9e-b679-6cb9b695649b)

![image](https://github.com/nweerasuriya/Python_Upskilling/assets/65176466/8b5a73c9-d22f-42d3-a5ff-b9a35f8d6ddb)





