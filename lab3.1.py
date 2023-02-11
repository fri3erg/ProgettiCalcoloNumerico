import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pandas as pd

" ----------- Esercizio 1 Approssimazione di un set di dati tramite Minini quadrati ----------- "

x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.50])

N = x.size 

for n in (1,2,3,5,7): #grado polinomio
    
    " Costruzione matrice A "
    A = np.zeros((N, n+1))
    for i in range(n+1):
        A[:,i] = x**i
    print("A:\n", A, "\n")

    " Risoluzione tramite equazioni normali ATAx = ATy "
    ATA = np.dot(A.T, A )
    ATy = np.dot(A.T, y)
    
    L = scipy.linalg.cholesky(ATA, lower=True)
    alpha1 = scipy.linalg.solve(L,ATy, lower=True) 
    alpha_normali = np.linalg.solve(L.T,alpha1)  
    
    print("alpha_normali = ", alpha_normali)

    " Risoluzione tramite SVD"
    U, s, Vh  = scipy.linalg.svd(A)
    alpha_svd = np.zeros(s.shape)
    
    for j in range(n+1):
        uj = U[:,j] 
        vj = Vh[j,:]
        alpha_svd = alpha_svd+((uj @ y) * vj ) / s[j]


    " Funzione per valutare polinomio p(x) "
    def p(alpha, x):
        A = np.zeros((x.size,len(alpha)))
        for i in range(len(alpha)):
          A[:,i] = x**i
        
        y = np.dot(A,alpha)
        return y

    " Grafici con confronti "
    
    x_plot = np.linspace(1,3,100)
    y_normali = p(alpha_normali, x_plot)
    y_svd = p(alpha_svd, x_plot)

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1) 
    plt.plot(x,y,'o', color='blue')
    plt.plot(x_plot,y_normali, color='red')
    plt.title('Approssimazione tramite Eq. Normali con n= %i'%n)

    plt.subplot(1, 2, 2)
    plt.plot(x,y,'o')
    plt.plot(x_plot,y_svd, color='red')
    plt.title('Approssimazione tramite SVD con n= %i'%n)

    plt.show()
    
    " Confronto errori sui punti " 
    yEq = p(alpha_normali,x)
    ySVD = p(alpha_svd,x)
    
    errEq = np.linalg.norm (y-yEq, 2) 
    errSVD = np.linalg.norm (y-ySVD, 2) 
    print ('Errore di approssimazione con Eq. Normali: ', errEq,'\n')
    print ('Errore di approssimazione con SVD: ', errSVD, '\n')
    



" ----------- Esercizio 2 Approssimazione di un set csv tramite Minini quadrati ----------- "


data = pd.read_csv("HeightVsWeight.csv")
data = np.array(data)
print("data.shape: ", data.shape)

for n in (1,2,3,5,7):
    

    x = data[:, 0] 
    y = data[:, 1]   
    N = x.size  

    A = np.zeros((N,n+1))

    for i in range(n+1):
        A[:,i]=x**i

    " Risoluzione tramite equazioni normali "
    ATA = np.matmul(A.T,A)
    ATy = np.matmul(A.T,y)

    L = scipy.linalg.cholesky(ATA,lower=True)
    alpha1 = scipy.linalg.solve(L,ATy)
    alpha_normali = scipy.linalg.solve(L.T,alpha1)

    " Risoluzione tramite SVD "
    U, s, Vh = scipy.linalg.svd(A)
    alpha_svd = np.zeros(s.shape)
    for j in range(n+1):
        uj=U[:,j]
        vj=Vh[j,:]
        alpha_svd=alpha_svd+(np.dot(uj,y)*vj)/s[j]
    

    " Grafici dei risultati "
    
    def p(alpha, x):
        A=np.zeros((x.size,len(alpha)))
        for i in range(len(alpha)):
            A[:,i]=x**i
        y = np.dot(A,alpha)
        return y

    x_plot = np.linspace(min(x),max(x),100)
    y_normali = p(alpha_normali, x_plot)
    y_svd=p(alpha_svd,x_plot)

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(x,y,'.', color='blue')
    plt.plot(x_plot,y_normali, color='red')
    plt.title('Approssimazione tramite Eq. Normali con n= %i '%n)

    plt.subplot(1, 2, 2)
    plt.plot(x,y,'.', color='blue')
    plt.plot(x_plot,y_svd, color='red')
    plt.title('Approssimazione tramite SVD con n= %i '%n)
    plt.show()