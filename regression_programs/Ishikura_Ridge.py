# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

class ridge_regression():
    def __init__(self,lamda=1,func="linear",func_para=1,s=1):
        self.lamda = lamda
        self.func = func
        self.func_para = func_para
        self.s = s
    def poly_basis(self,X_poly):
        n_p,d_p= X_poly.shape
        phi_p = np.empty((n_p,d_p*self.func_para))
        for i in range(self.func_para):
            phi_p[:,d_p*i:d_p*(i+1)] = np.power(X_poly,i+1)
        return phi_p
    def gauss_basis(self,X_gauss):
        n_g,d_g = X_gauss.shape
        phi_g = np.empty((n_g,d_g*self.func_para))
        for i in range(self.func_para):
            mu = (i+1)/self.func_para
            phi_g[:,d_g*i:d_g*(i+1)] = np.exp(-(X_gauss-mu)**2/(2**self.s))
        return phi_g
    def fit(self,X,t):
        n = X.shape[0]
        if self.func == "poly":
            X = self.poly_basis(X)
        elif self.func == "gauss":
            X = self.gauss_basis(X)
        X = np.hstack([np.ones((n,1)),X])
        
        A = X.T @ X+self.lamda*np.eye(X.shape[1])
        A[0,0] -=self.lamda
        w = np.linalg.inv(A) @ X.T @ t
        #w = np.linalg.inv(X.T @ X+self.lamda*np.identity(X.shape[1])) @ X.T @ t
        self.w0_ = w[0]
        self.w_ = w[1:]
        return self
    def predict(self,X_test):
        if self.func == "poly":
            X_test = self.poly_basis(X_test)
        elif self.func == "gauss":
            X_test = self.gauss_basis(X_test)
        y = np.dot(X_test,self.w_) + self.w0_
        return y

def MSE(t,y):
    return np.mean((t-y)**2)