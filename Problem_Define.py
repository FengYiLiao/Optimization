import numpy as np

'''
    Benchmark function  F: unconstrained problem
                        g: Consrtraint   
'''

def F1(x):
    '''Dim = 30, SearchRange=[-100,100]'''
    X = np.dot(x,x.T)
    y = X.diagonal() #左上至右下對角線
    return y

def F2(x):
    '''Dim = 30, SearchRange=[-10,10]'''
    A = np.abs(x)
    B = np.sum(A,axis=1)
    C = 1
    for i in range(x.shape[1]):
        C = C * A[:,i]
    F = B + C
    return F

def F3(x):
    '''Dim = 30, SearchRange=[-100,100]'''
    F = np.zeros([x.shape[0]])
    for i in range(x.shape[1]):
        C = np.sum(x[:,0:i+1],axis=1)
        C = C**2
        F = F + C
    return F

def F4(x):
    '''Dim = 30, SearchRange=[-100,100]'''
    F = np.max(abs(x),axis=1)
    return F

def F5(x):
    '''Dim = 30, SearchRange=[-30,30]'''
    Y = 0
    for i in range(x.shape[1]-1):
        Y = Y + 100*(x[:,i+1]-x[:,i]**2)**2+(x[:,i]-1)**2
    return Y

def F6(x):
    '''Dim = 30, SearchRange=[-100,100]'''
    X = abs(x+0.5)**2
    F = X.sum(axis=1)
    return F

def F7(x):
    '''Dim = 30, SearchRange=[-1.28,1.28]'''
    Y = 0
    for i in range(x.shape[1]):
        Y = Y + (i+1)*x[:,i]**4
    return Y + np.random.random()

def F8(x):
    '''Dim = 30, SearchRange=[-500,500]'''
    Y = -np.sin(np.sqrt(np.abs(x)))*(x)
    F = Y.sum(axis=1)
    return  F
def F9(x):
    '''Dim = 30, SearchRange=[-5.12,5.12]'''
    A = x**2
    B = -10*np.cos(2*np.pi*x)
    C = 10
    Y = A+B+C
    return Y.sum(axis=1)

def F10(x):
    '''Dim = 30, SearchRange=[-32,32]'''
    n = x.shape[1]
    Y = -20*np.exp(-0.2*np.sqrt(np.sum(x**2,axis=1)/n))-np.exp(np.sum(np.cos(2*np.pi*x),axis = 1)/n)+20 + np.e
    return Y

def F11(x):
    '''Dim = 30, SearchRange=[-600,-600]'''
    B = 1
    for i in range(x.shape[1]):
        B = B*np.cos(x[:,i]/np.sqrt(i+1))
    Y = np.sum(x**2 ,axis = 1)/4000 - B + 1
    return Y

def F12(x):
    '''Dim = 30, SearchRange=[-50,50]'''
    n = x.shape[1]
    a = 10
    k = 100
    m = 4
    y = 1 + (x+1)/4
    u1 = np.where(x>a,k*(x-a)**m,0)
    u2 = np.where(x<-a,k*(-x-a)**m,0)
    u = u1+u2
    F = np.pi/n*(10*np.sin(np.pi*y[:,0])+np.sum((y[:,:-1]-1)**2*(1+10*np.sin(np.pi*y[:,1:])**2),axis=1)+(y[:,-1]-1)**2)+np.sum(u,axis=1)
    return F

def F13(x):
    '''Dim = 30, SearchRange=[-50,50]'''
    a = 5
    k = 100
    m = 4
    u1 = np.where(x>a,k*(x-a)**m,0)
    u2 = np.where(x<-a,k*(-x-a)**m,0)
    u = u1 + u2
    F = 0.1*(np.sin(3*np.pi*x[:,0])**2+np.sum((x-1)**2*(1+np.sin(3*np.pi*x+1)),axis = 1)+(x[:,-1]-1)**2*(1+np.sin(2*np.pi*x[:,-1])**2))+np.sum(u,axis=1)
    return F

def h(x):#equality constraint
    h = np.zeros((len(x),1))
    h[:,0] = x[:,0]-x[:,1]+1.0
    return  h


def g(x):
    g = np.zeros((len(x), 1))
    g[:, 0] = x[:, 28] + x[:, 29] + 20
    return g



