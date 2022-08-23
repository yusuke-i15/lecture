# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 12:23:30 2022

@author: 81702
"""

import numpy as np
from scipy.stats import multivariate_normal 
from scipy.special import digamma,logsumexp

#VBアルゴリズムを行うクラス
class VB_GMM():
    def __init__(self,k=4,tol=0.001,max_iter=1000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
    def set_params(self,X):
        #パラメータの初期値などの設定
        self.N,self.d = X.shape
        self.alpha0 = 0.1
        self.beta0 = 1
        self.v0 = self.d
        self.m0 = np.random.randn(self.d)
        self.W0 = np.identity(self.d)
        self.W0_inv = np.linalg.inv(self.W0)
        self.alpha = np.full(self.k,self.alpha0)
        self.beta = np.full(self.k,self.beta0)
        self.v = np.full(self.k,self.v0)
        self.m = np.random.randn(self.k,self.d)
        self.W = np.tile(self.W0,(self.k,1,1))
    def calc_gmm(self,X):
        #XについてGMMの対数尤度の計算　gmm:N×k    
        gmm = np.empty((self.N,self.k))
        pi = self.alpha/(np.sum(self.alpha))
        for i in range(self.k):
            gmm[:,i] = pi[i]*multivariate_normal.pdf(X,mean=self.m[i],cov=(self.v[:,None,None]*self.W)[i])
        return gmm
    def E_step(self,X):
        #Eステップの実行 r_nkの計算
        log_pi = digamma(self.alpha)-digamma(np.sum(self.alpha))
        log_sigma = np.sum([digamma((self.v-i)/2) for i in range(self.d)])+self.d*np.log(2)+np.log(np.linalg.det(self.W))
        temp_array = np.empty((self.N,self.k))
        
        for i in range(self.N):
            for j in range(self.k):
                temp_diff = (X[i,:]-self.m[j,:]).reshape((self.d,1))
                temp_array[i,j] = temp_diff.T @ self.W[j,:,:] @ temp_diff
        log_rho = log_pi+0.5*log_sigma-self.d/(2*self.beta)-self.v*temp_array/2
        log_r = log_rho - logsumexp(log_rho,axis=1,keepdims=True)
        self.r_nk = np.exp(log_r)
    def M_step(self,X):
        #Mステップの実行　パラメータalpha,beta,v,m,Wの更新
        S1_k = np.sum(self.r_nk,axis=0) #(,k)
        Sx_k = np.dot(self.r_nk.T,X) #(k,d)
        Sxxt_k = np.empty((self.k,self.d,self.d))
        for i in range(self.k):
            Sxxt_k[i,:,:] = np.dot(self.r_nk[:,i]*X.T,X)
        self.alpha = self.alpha0 + S1_k
        self.beta = self.beta0 + S1_k
        self.v = self.v0 + S1_k
        self.m = (self.beta0*self.m0 +Sx_k)/self.beta[:,None]
        temp_m0 = self.m0.reshape(1,self.d)
        m0tm0 =np.dot(temp_m0.T,temp_m0)
        for i in range(self.k):
            mk = self.m[i].reshape(1,self.d)
            W_inv = self.W0_inv + m0tm0 + Sxxt_k[i,:,:]-self.beta[i]*np.dot(mk.T,mk)
            self.W[i,:,:] = np.linalg.pinv(W_inv)
    def EM(self,X):
        self.set_params(X)
        log_likelihood_mean = np.mean(np.log(np.sum(self.calc_gmm(X),axis=1)))
        print("iter 0,log likelihood:",np.sum(np.log(np.sum(self.calc_gmm(X),axis=1))))
        for i in range(self.max_iter):
            self.E_step(X)
            self.M_step(X)
            temp = log_likelihood_mean
            log_likelihood_mean = np.mean(np.log(np.sum(self.calc_gmm(X),axis=1)))
            diff = abs(log_likelihood_mean - temp)
            print("iter:{},log_likelihood:{},mean_diff:{}".format(i+1,np.sum(np.log(np.sum(self.calc_gmm(X),axis=1))),diff))
            if diff <= self.tol:#対数尤度の平均の差が閾値よりも小さくなったら終了
                print("finished,iter:",i+1)
                return self.r_nk
        print("not converged ",self.max_iter)
        return self.r_nk