############################################################
### Parameters

# How many observations
numObservations = 1000

# Intercept and coefficent
intercept = 45
x1coef = 0.15

# The variance on the response variable
variance = 100

############################################################
### Functions

# The linear function that we'll reconstruct
f <- function(x) { intercept + x1coef*x }

# A function to add the Gaussian error to the output of f
g <- function(x) { rnorm(1,f(x),variance) }

############################################################
### Create data

# Generate random x values
xvals = floor(runif(numObservations,0,10000))

# Given the x values, generate y values according to g
yvals = sapply(xvals,g)


############################################################
### Check the fit of a model to the data

# Do linear regression on the generated data
lrfit <- glm(yvals ~ xvals)

# Print the summary of the model
lrfit

# Print the difference from the fit values to the ones used to generate
interceptFit = lrfit$coef[1]
abs(interceptFit - intercept)
x1coefFit = lrfit$coef[2]
abs(x1coefFit - x1coef)

# Plot the data and the fit model
plot(yvals ~ xvals)
abline(lrfit,col="blue")

