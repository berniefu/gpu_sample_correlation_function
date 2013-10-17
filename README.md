This code uses GPU to implement the sampling method to the Correlation Function problem.

The problem is given more details in http://www.pdl.cmu.edu/PDL-FTP/AstroDISC/astro08-fu.pdf

Basically, we sample a small number of objects, calculate their Correlation Function, and then repeat multiple times to
calculate the mean and standard deviation, and further estimate sampling errors.

We had similar sequential implementations, one in Hadoop, and one in OpenMP 
(http://www.pdl.cmu.edu/AstroDISC/disc-dist-code.shtml)

In terms of this code, we argue that the parallelism of GPU now can be much higher than that of CPU. This code made the
first try to explore the powerful GPU modules. Testing on a EC2 module with GPU, the parallel code is comparable to
a CPU solution on a 128 CPU cores.

Please contact berniefu@gmail.com for more details.
