# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:45:20 2022

@author: 81702
"""
#GMMでのEMアルゴリズムを行うクラス

import numpy as np
from scipy.stats import multivariate_normal 

class EM_GMM():
    def __init__(self,k=1,tol=0.001,max_iter=1000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
    def set_params(self,X):
        #パラメータの初期値などの設定
        self.N,self.d = X.shape
        self.mu = np.random.randn(self.k,self.d)
        self.sigma = np.tile(np.identity(self.d),(self.k,1,1))
        self.pi = np.ones(self.k)/self.k
    def calc_gmm(self,X):
        #XについてGMMの確率密度関数の計算　gmm:N×k
        #\pi_k*N(x_n|\mu_k,\sigma_k)
        gmm = np.empty((self.N,self.k))
        for i in range(self.k):
            gmm[:,i] = self.pi[i]*multivariate_normal.pdf(X,mean=self.mu[i],cov=self.sigma[i])
        return gmm
    def E_step(self,X):
        #Eステップの実行 負担率r_nkの計算
        gmm_pdf = self.calc_gmm(X) #\pi_k*N(x_n|\mu_k,\sigma_k)の計算
        self.r_nk = gmm_pdf/np.sum(gmm_pdf,axis=1).reshape((self.N,1)) #(N,k)
    def M_step(self,X):
        #Mステップの実行　パラメータmu,sigma,piの更新
        S1_k = np.sum(self.r_nk,axis=0) #(,k)
        self.pi = S1_k/self.N
        self.mu = np.dot(self.r_nk.T,X)/S1_k.reshape((self.k,1)) #(k,d)
        for i in range(self.k):
            temp_array = X-self.mu[i,:]
            self.sigma[i,:,:] = np.dot(self.r_nk[:,i]*temp_array.T,temp_array)/S1_k[i]
    def EM(self,X):
        self.set_params(X)
        log_likelihood_mean = np.mean(np.log(np.sum(self.calc_gmm(X),axis=1)))
        print("iter 0,log likelihood:",np.sum(np.log(np.sum(self.calc_gmm(X),axis=1))))
        for i in range(self.max_iter):
            self.E_step(X)
            self.M_step(X)
            temp = log_likelihood_mean
            log_likelihood_mean = np.mean(np.log(np.sum(self.calc_gmm(X),axis=1)))
            diff = np.abs(log_likelihood_mean - temp)
            print("iter:{},log_likelihood:{},diff:{}".format(i+1,np.sum(np.log(np.sum(self.calc_gmm(X),axis=1))),diff))
            if diff <= self.tol:#対数尤度の平均の差が閾値よりも小さくなったら終了
                print("finished,iter:",i+1)
                return self.r_nk
        print("not converged ",self.max_iter)
        return self.r_nk