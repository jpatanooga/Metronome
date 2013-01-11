############################################################
### Parameters

# How many observations
numObservations = 1000

# Set the maximum number of values we can choose (randomly)
# for x. Chosen to be 1/4 of the number of observations.
maxValueForX = floor(numObservations/4)

# Intercept and coefficent
intercept = 45
x1coef = 0.15

# The variance on the response variable
variance = 100

# The filename to use. Will not print if NULL
outputFilename = "lrdata.txt"

# Use command line args if given
args = commandArgs(TRUE)
if (length(args) == 5) {
  numObservations = as.numeric(args[1])
  intercept = as.numeric(args[2])
  x1coef = as.numeric(args[3])
  variance = as.numeric(args[4])
  outputFilename = args[5]
}

############################################################
### Functions

# The linear function that we'll reconstruct
f <- function(x) { intercept + x1coef*x }

# A function to add the Gaussian error to the output of f
g <- function(x) { rnorm(1,f(x),variance) }

############################################################
### Create data


############################################################
### Create and output the data
output = file(outputFilename,"w")
blocksize = 1000
numblocks = numObservations/blocksize
for (i in 1:numblocks) {

  # Generate random x values
  xvals = floor(runif(blocksize,0,maxValueForX))

  # Given the x values, generate y values according to g
  yvals = sapply(xvals,g)

  for (j in 1:blocksize) {
    cat(yvals[j]," |f 0:", xvals[j],"\n",sep="",file=output)
  }
  flush(output)
}
close(output)
