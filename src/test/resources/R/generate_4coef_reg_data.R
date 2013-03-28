############################################################
### Parameters

# How many observations
numObservations = 100

# Set the maximum number of values we can choose (randomly)
# for x. Chosen to be 1/4 of the number of observations.
maxValueForX = floor(numObservations/10)

# Intercept and coefficent

intercept = 0

x1coef = 1

x2coef = 2

x3coef = 3

x4coef = 4


# The variance on the response variable
variance = 10

# The filename to use. Will not print if NULL
outputFilename = "lrdata.txt"

# Use command line args if given
args = commandArgs(TRUE)
if (length(args) == 3) {
  numObservations = as.numeric(args[1])
#  intercept = as.numeric(args[2])
#  x1coef = as.numeric(args[3])
  variance = as.numeric(args[2])
  outputFilename = args[3]
}

############################################################
### Functions

# The linear function that we'll reconstruct
f <- function(x1, x2, x3, x4) { intercept + x1coef*x1 + x2coef*x2 + x3coef*x3 + x4coef*x4 }

# A function to add the Gaussian error to the output of f
g <- function(x1, x2, x3, x4) { rnorm(1,f(x1, x2, x3, x4),variance) }

############################################################
### Create data


############################################################
### Create and output the data
output = file(outputFilename,"w")
### blocksize = 1000
### numblocks = numObservations/blocksize

print(numObservations)

#for (i in 1: numObservations) {

  # Generate random x values
  x1vals = floor(runif(numObservations,0,maxValueForX))
  x2vals = floor(runif(numObservations,0,maxValueForX))
  x3vals = floor(runif(numObservations,0,maxValueForX))
  x4vals = floor(runif(numObservations,0,maxValueForX))

  # Given the x values, generate y values according to g
#  yvals = sapply(x1vals,x2vals,x3vals,x4vals,g)

x1vals

x2vals

  for (j in 1: numObservations) {
#  	y = g(x1vals,x2vals,x3vals,x4vals)
#    cat(yvals[j]," |f 0:", x1vals[j]," 1:", x2vals[j]," 2:", x3vals[j]," 3:", x4vals[j],"\n",sep="",file=output)
    cat(g(x1vals[j],x2vals[j],x3vals[j],x4vals[j])," |f 0:", x1vals[j]," 1:", x2vals[j]," 2:", x3vals[j]," 3:", x4vals[j],"\n",sep="",file=output)
  }
  flush(output)
#}
close(output)
