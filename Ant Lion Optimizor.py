import numpy as np
import pandas as pd
import Problem_Define as P
import matplotlib.pyplot as plt
'''
    Lastest Update: July 7, 2021
    Ant Lion Optimizor(ALO) is implemented by F.Y LIAO
    ALO is published by Seyedali Mirjalili in 2015. 
    https://www.sciencedirect.com/science/article/abs/pii/S0965997815000113
    
    Note:
    ALO can be used to solve both unconstrained and constrained problems.
    Input functions should be in the form of multiple scales (Calculate multiple function values once).   
    
    Minamize:
            f(x)
        subject to
            hi(x) = 0; i = 0 to p
            gi(x) <= 0; j = 1 to m
            
    Usage
        ALO(ObjFun,Population,Dimension,SearchRange,Constrain)
        Pamareters
            ObjFun: objection function in the form of multiple scales (Calculate multiple function values once). 
            Population: number of population 
            Dimension: dimension of the objection function(number of variables)
            SearchRange: searchrange in the form of [Lowerbound, Upperbound]
            Constrain: equatity and inequality. in the form of JSON. 
    Example:
        Unconstrained:
            Minamize:
                f(x) = x1*x1 + x2*x2 + ... + x30*x30
            SearchRange
               -50<=X<=50, X=[x1,x2,...x30]
                
            ALO(ObjFun=F1,Population = 30,Dimension=30,SearchRange=[-50,50])
            
            def F1(x):
                X = np.dot(x,x.T)
                y = X.diagonal() 
                return y

        Constrained(equality):
            Minamize:
                f(x) = x1*x1 + x2*x2 + ... + x30*x30
            Subject to
                x0 - x1 + 1 = 0 
            SearchRange
               -50<=X<=50, X=[x1,x2,...x30]
                
            ALO(ObjFun=F1,Population = 30,Dimension=2,SearchRange=[-50,50],Constrain={'eq':h})
            
            def h(x):
                h = np.zeros((len(x),1))
                h[:,0] = x[:,0]-x[:,1]+1.0
                return  h
        
        Constrained(inequality):
            Minamize:
                f(x) = x1*x1 + x2*x2 + ... + x30*x30
            Subject to
                x0 - x1 + 1 <= 0
            SearchRange
               -50<=X<=50, X=[x1,x2,...x30]
            
            ALO(ObjFun=P.F1,Population = 30,Dimension=2,SearchRange=[-50,50],Constrain={'ineq':P.g})
            
            def g(x):
                g = np.zeros((len(x),1))
                g[:,0] = x[:,0]-x[:,1]+1.0
                return  g
                
        Constrained(both equality and inquality):
           Minamize:
                f(x) = x1*x1 + x2*x2 + ... + x30*x30
            Subject to
                x0 - x1 + 1 = 0 
                x29 + x30 + 20 <= 0
            SearchRange
               -50<=X<=50, X=[x1,x2,...x30]
                
            ALO(ObjFun=F1,Population = 30,Dimension=2,SearchRange=[-50,50],Constrain={'eq':h,'ineq':g})
            
            def h(x):
                h = np.zeros((len(x),1))
                h[:,0] = x[:,0]-x[:,1]+1.0
                return  h      
            def g(x):
                g = np.zeros((len(x),1))
                g[:,0] = x[:,28] + x[:,29] + 20
                return  g      
'''

Maxiteration = 500
W = np.array([2,2,3,4,5,6])
ratio = np.array([0,0.1,0.5,0.75,0.9,0.95])
Pop = 0

def fitness(C):
    Fitness = C.copy()
    if Fitness.min() <= 0: #maintain the fitness positive
        Fitness = Fitness - Fitness.min() + 100
    Fitness = 1.0/Fitness #the lower function value, the higher fitness value
    return Fitness
def RouletteWheel(C):#Input is a 1-D array
    Fitness = fitness(C)
    Fitness = Fitness/Fitness.sum()#
    X = np.cumsum(Fitness)
    r = np.random.random()
    j = 0
    for i in range(X.shape[0]):
        if r < X[i]:
            j = i
            break
    return j

CC = []
DD = []
k = 1
def UpdateCandD(C,d,t,T,Antlion):
    L = C.copy()
    U = d.copy()
    w =  W[t > ratio*T].max()
    I = pow(10,w)*t/T
    C = C/I
    d = d/I
    if C[0] < L[0]:
        C = L.copy()
    if d[0] > U[0]:
        d = U.copy()
    C = Antlion + C
    d = Antlion + d
    return np.array([C,d])

def randonwalk(D,L,U,iter):
    step = np.zeros([Pop,D])
    for j in range(Pop):
        rand = np.random.random([iter,D])
        X = np.where(rand>0.5,1,-1)
        Y = np.cumsum(X,axis=0)
        step[j] = Y[iter-1]

    X = normalize(step,L,U,iter)
    return X

def normalize(X,L,U,iter):
    a = X.min(axis=0)#find min for each row
    b = X.max(axis=0)#find max for each row
    B = (X - a)*(U-L)/(b-a)+L
    return B


def PshenichnyFitness(M,ObjFun,Constrain):
    R = 10 #Penalty
    noConstrain = 0
    noEq = 0
    noIneq = 0
    if "eq" in Constrain.keys():
        h = Constrain["eq"]
        H = h(M)
        noConstrain += H.shape[1]
        noEq += H.shape[1]
        H = np.abs(H)
    if "ineq" in Constrain.keys():
        g = Constrain["ineq"]
        G = g(M)
        noConstrain += G.shape[1]
        noIneq += G.shape[1]
        G = np.where(G>0,G,0)

    F = ObjFun(M)
    if noIneq > 0 and noEq > 0:
        Violent = np.hstack((H.reshape(Pop,noEq),G.reshape(Pop,noIneq)))
    elif noEq >0 and noIneq ==0:
        Violent = H.reshape(Pop, noEq)
    elif noIneq > 0 and noEq==0 :
        Violent = G.reshape(Pop, noIneq)

    if Violent.ndim >1:
        Violent = Violent.max(axis=1)
    P = F.reshape(Pop,1)+R* Violent.reshape(Pop,1)
    return P
def ALO(ObjFun,Population,Dimension,SearchRange,Constrain = {}):
    global Pop
    Pop = Population
    D = Dimension
    PopAnt = Population
    PopAntlion = Population
    LowerBound = np.zeros([D])
    UpperBound = np.zeros([D])
    for i in range(D):
        LowerBound[i] =  SearchRange[0]
        UpperBound[i] = SearchRange[1]
    # np.random.seed(0)
    Mant = np.random.uniform(LowerBound,UpperBound,[PopAnt,D])
    Mantlion = np.random.uniform(LowerBound,UpperBound,[PopAntlion,D])
    if Constrain == {}:#unconstrain problem
        MOA = ObjFun(Mant)
        MOAL = ObjFun(Mantlion)
    else:#constrain problem
        MOA = PshenichnyFitness(Mant,ObjFun,Constrain)
        MOAL = PshenichnyFitness(Mantlion,ObjFun,Constrain)
    '''Select the elite '''
    elite = np.hstack((Mantlion,MOAL.reshape(PopAntlion,1)))
    Order = np.argsort(elite[:,D])
    elite = elite[Order]
    elite = elite[0]
    i = 1

    Cost = []
    # SearchHistoryX =[]
    # SearchHistoryY = []
    while i < Maxiteration:
        SelectedAntlion = np.zeros([PopAnt,D])
        index = []
        if i == 50:
            print(i)
        for j in range(PopAnt):
            r = RouletteWheel(MOAL)
            index.append(r)
            SelectedAntlion[j] = Mantlion[r].copy()
        LA,UA = UpdateCandD(LowerBound,UpperBound,i,Maxiteration,SelectedAntlion)
        RA = randonwalk(D,LA,UA,iter = i)

        LE, UE = UpdateCandD(LowerBound, UpperBound, i, Maxiteration, elite[:-1])
        RE = randonwalk(D,LE,UE,iter = i )

        Mant = (RA+RE)/2
        '''超過upperbound and lowerbound'''
        Mant = np.where(Mant>UpperBound[0],UpperBound[0],Mant)
        Mant = np.where(Mant<LowerBound[0],LowerBound[0],Mant)

        if Constrain == {}:
            MOA = ObjFun(Mant)
            # MOAL = ObjFun(Mantlion)
        else:
            MOA = PshenichnyFitness(Mant, ObjFun, Constrain)
            # MOA = PshenichnyFitness(Mantlion, ObjFun, Constrain)

        M_combined_antlion = np.hstack((Mantlion,MOAL.reshape(PopAntlion,1)))
        M_combined_ant = np.hstack((Mant,MOA.reshape(PopAnt,1)))
        M_combined = np.vstack((M_combined_antlion,M_combined_ant))
        Order_com = np.argsort(M_combined[:,D])
        M_combined = M_combined[Order_com]
        z = Pop
        Mantlion = M_combined[0:Pop,:-1].copy()

        MOAL = ObjFun(Mantlion)

        E = np.hstack((Mantlion[0],MOAL[0]))

        if E[D] < elite[D]:
            elite = E.copy()
        i = i + 1
        print("Iteration ={:d} , Cost ={:f}".format(i,elite[D]))

        # SearchHistoryX.append(Mantlion[:,0].flatten())
        # SearchHistoryY.append(Mantlion[:,1].flatten())

        Cost.append(elite[D])
    # plt.subplot(122)
    # plt.plot(SearchHistoryX, SearchHistoryY, 'ro')
    # plt.xlim(SearchRange[0],SearchRange[1])
    # plt.ylim(SearchRange[0],SearchRange[1])
    # plt.xlabel("x1")
    # plt.ylabel("x2")
    # plt.title("Search History")
    # plt.subplot(121)
    # plt.plot(range(1,Maxiteration),Cost)
    # plt.xlabel("iteration")
    # plt.ylabel("Cost")
    # plt.title("Covergence Curve")
    # plt.show()
    # plt.plot(range(1, Maxiteration), DD)
    # plt.plot(range(1,Maxiteration),CC)
    # plt.ylim(SearchRange[0], SearchRange[1])
    # plt.xlim(1,Maxiteration)
    # plt.legend(["d","c"])
    # plt.show()

    print("Solution")
    print(elite)

if __name__ == '__main__':
    ALO(ObjFun=P.F1,Population = 30,Dimension=30,SearchRange=[-50,50],Constrain={'eq':P.h})
