# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:19:28 2022

@author: 81702
"""

import numpy as np
import matplotlib.pyplot as plt

trial = 100
K = 8
d = 2
sigma = 1
lam = 1
max_T = 10000
ai = np.empty((d,K))
theta_star = np.array([3,1])
for i in range(K):
    ai[:,i] = [np.cos(np.pi*(i+1)/4),np.sin(np.pi*(i+1)/4)]
r_star = np.max(np.dot(theta_star,ai))

rewards = np.zeros((trial,max_T))
for j in range(trial):
    A = lam*np.identity(d)
    b = np.zeros((d,1))
    theta_hat = np.zeros((d,1))
    UCBi = np.zeros(K)
    for t in range(1,max_T+1):
        A_inv = np.linalg.inv(A)
        theta_hat = np.dot(A_inv,b)
        alpha_t = np.sqrt(2*np.log2(t))
        for i in range(K):
            ait = ai[:,i].reshape(d,1)
            UCBi[i] = np.dot(ait.T,theta_hat) + sigma*alpha_t*np.sqrt(ait.T @ A_inv @ ait)
        max_i = np.argmax(UCBi)
        ais = ai[:,max_i].reshape(d,1)
        rewards[j,t-1] = np.dot(theta_star,ais)
        r = np.random.normal(rewards[j,t-1],1)
        A += np.dot(ais,ais.T)
        b += r*ais

regret = (r_star - np.average(rewards,axis=0))

plt.plot(regret.cumsum())
plt.title("Cumulative Regret")