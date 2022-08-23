EM_GMM.py:This includes EM algorithm class
VB_GMM.py:This includes GM algorithm class

EM_GMMexe.py:execute EM algorithm
   example:
        python EM_GMMexe.py x.csv z.csv params.dat
	number of cluster k = ?
	max_iter = ?
   x.csv:input data file
   z.csv:output file name which include posterior probabilities of z_n
   params.dat:output parameters file name which include parameters
   please enter the k for example 4 and max_iter for example 1000
     show the value of the log likelihood and mean log likelihood difference at each iteration
   and plot the result in 3D
VB_GMMexe.py:execute VB algorithm
   example:
        python VB_GMMexe.py x.csv z.csv params.dat
	number of cluster k = ?
	max_iter = ?
   x.csv:input data file
   z.csv:output file name which include posterior probabilities of z_n
   params.dat:output parameters file name which include parameters
   please enter the k for example 4 and max_iter for example 1000
     show the value of the log likelihood and mean log likelihood difference at each iteration
   and plot the result in 3D  