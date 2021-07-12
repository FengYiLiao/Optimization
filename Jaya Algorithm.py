import numpy as np
import Problem_Define as P
import random
import pandas as pd
'''
    Lastest Update: July 7, 2021
    Jaya Algorithm is implemented by F.Y Liao using Python.
    Jaya Algorithm is published by Ravipudi Venkata Rao in 2016
    http://www.growingscience.com/ijiec/Vol7/IJIEC_2015_32.pdf. 
    Function: Solve unconstrained problem.
    
    Minamize:
            f(x)

    Usage:
        Jaya(ObjFun,Dimension,Population,SearchRange)
    
    Pamareters:
            ObjFun: objection function in the form of multiple scales (Calculate multiple function values once). 
            Population: number of population 
            Dimension: dimension of the objection function(number of variables)   
            SearchRange: searchrange in the form of [Lowerbound, Upperbound]

     Example:
            Minamize:
                f(x) = x1*x1 + x2*x2 
            SearchRange
               -50<=X<=50, X=[x1,x2]
                
            Jaya(ObjFun=testf,Population=10,Dimension=2,SearchRange=[-50,50])
            
            def testf(x):
                return x[:,0]*x[:,0]*+x[:,1]*x[:,1]
'''


# ObjFun,Constrain,Population,epsilon
MaxIteration = 1000
ftol = 1.0e-8
def Jaya(ObjFun,Population,Dimension,SearchRange):
    n = Dimension #number of design variables
    P = Population #number of candidates = number of population
    '''Generate (Population*number of design variables) initail guess for First iteration '''
    x = np.zeros((P,n))
    random.seed(0)
    for k in range(P):
        for j in range(n):
            x[k,j] = random.uniform(SearchRange[0],SearchRange[1])
    '''Calculate value of objective function'''
    F = ObjFun(x)
    F = F.reshape((F.shape[0],1))
    '''Construct a matrix that have x and f'''
    X = np.hstack((x,F))
    '''Find out best candidate and worst candidate'''
    Order = np.argsort(X[:,n])#j對value of objfun column sort, return indices that would sort an array
    X = X[Order]
    WorstX = X[P-1,:-1]
    BestX = X[0,:-1]
    PreF = X[0,n]
    NowF = PreF
    Xprime = np.zeros((P,n+1))
    '''construct random number for design variables'''
    R1 = np.zeros(P)
    R2 = np.zeros(P)
    for i in range(MaxIteration):
        '''construct random number for design variables'''
        for j in range(n):
            for k in range(P):
                R1[k] = random.random()
                R2[k] = random.random()
            Xprime[:,j] = X[:,j] + R1*(BestX[j]-np.absolute(X[:,j])) - R2*(WorstX[j]-np.absolute(X[:,j]))

        F = ObjFun(Xprime)
        # Fprime = ObjFun(Xprime)
        Xprime[:,n] = F.flatten()
        '''update design variables,
           rule: if (F[n+1]<F[n]) 
                     update. else re
                 else
                     remain the same'''
        for k in range(P):
            if F[k] < X[k,n]:
                X[k,:] = Xprime[k,:]
        Order = np.argsort(X[:, n])  # j對value of objfun column sort, return indices that would sort an array
        X = X[Order]
        WorstX = X[P - 1, :-1]
        BestX = X[0, :-1]
        NowF = X[0,n]
        print("Iteration = {:f}. X = {}. ObjFun = {:f}".format(i+1,BestX,NowF))
        result = {"Design": BestX,"Function_Value":NowF}
        if NowF == np.inf or NowF == -np.inf:
            print("Diverge. No Solution.")
            break
        PreF = X[0,n]
    return result

def testf(x):
    return x[:,0]*x[:,0]*+x[:,1]*x[:,1]

if __name__ == '__main__':
    res = Jaya(ObjFun=testf,Population=10,Dimension=2,SearchRange=[-50,50])
    print(res)
