# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 16:53:46 2022

@author: 81702
"""

from VB_GMM import VB_GMM
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


data = sys.argv[1]
save_csv_r_nk = sys.argv[2]
save_dat_para = sys.argv[3]
X = np.loadtxt(data, delimiter=',')
"""
save_csv_r_nk = "z.csv"
save_dat_para = "params.dat"
X = np.loadtxt("x.csv", delimiter=',')
"""
print("number of cluster k =",end=" ")
k = int(input())
print("max_iter =",end=" ")
max_iter = int(input())
"""
k = 12
max_iter = 1000
"""
model = VB_GMM(k=k,max_iter=max_iter)
r_nk = model.EM(X)
y = np.argmax(r_nk,axis = 1)

#visualize
fig = plt.figure()
cm = plt.get_cmap("tab10")
ax = mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
for i in range(k):
    index = np.where(y==i)[0]
    ax.plot(X[index,0],X[index,1],X[index,2],"o",color = cm(i),ms=0.5)
ax.view_init(elev=30,azim=45)
plt.show()

#save Posterior probabilities in z.csv
np.savetxt(save_csv_r_nk,r_nk,delimiter=",")

#save parameters in params.dat
f = open(save_dat_para, "w", encoding="utf-8")
f.write("model : VB_GMM\n")
f.write("K = "+str(k))
f.write(",tol = "+str(model.tol))
f.write(",max_iter = "+str(max_iter)+"\n")
f.write("alpha\n")
f.write(str(model.alpha.ravel())+"\n")
f.write("beta\n")
f.write(str(model.beta.ravel())+"\n")
f.write("v\n")
f.write(str(model.v.ravel())+"\n")
f.write("m\n")
f.write(str(model.m.ravel())+"\n")
f.write("W\n")
f.write(str(model.W.ravel()))
f.close()
