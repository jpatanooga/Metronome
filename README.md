Metronome is a suite of parallel iterative algorithms that run natively on Hadoop's Next Generation YARN platform. 

* Based directly on work we did with [Knitting Boar](https://github.com/jpatanooga/KnittingBoar) and [IterativeReduce] (https://github.com/emsixteeen/IterativeReduce)
    * Parallel logistic regression
    * Scales linearly with input size
    * Built on top of BSP-style computation framework "Iterative Reduce" (Hadoop / YARN)
    * Uses Mahout's implementation of Stochastic Gradient Descent (SGD) as basis for worker process
* Linear Regression package can produce a linear regression model off large amounts of data
* Packaged in a new suite of parallel iterative algorithms called Metronome
    * 100% Java, ASF 2.0 Licensed, on github


# Original Presentation from Hadoop Summit EU 2013

<iframe src="http://www.slideshare.net/slideshow/embed_code/17636499" width="427" height="356" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC;border-width:1px 1px 0;margin-bottom:5px" allowfullscreen webkitallowfullscreen mozallowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="http://www.slideshare.net/jpatanooga/hadoop-summit-eu-2013-parallel-linear-regression-iterativereduce-and-yarn" title="Hadoop Summit EU 2013: Parallel Linear Regression, IterativeReduce, and YARN" target="_blank">Hadoop Summit EU 2013: Parallel Linear Regression, IterativeReduce, and YARN</a> </strong> from <strong><a href="http://www.slideshare.net/jpatanooga" target="_blank">Josh Patterson</a></strong> </div>




# Resources
* [IterativeReduce Programming Model] (https://github.com/emsixteeen/IterativeReduce/wiki/Iterative-Reduce-Programming-Guide)
* Using IRUnit - the IterativeReduce Unit Testing Framework
* [Running parallel linear regression with IRUnit on synthetic test data](https://github.com/jpatanooga/Metronome/wiki/Running-Parallel-Linear-Regression)

# Contributors and Special Thanks
* Michael Katzenellenbollen
* Dr. Jason Baldridge
* Dr. James Scott
