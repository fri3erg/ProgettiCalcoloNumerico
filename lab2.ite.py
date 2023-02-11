import numpy as np
import scipy
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
import matplotlib.pyplot as plt
import time

" ----------- Esercizio 2 Metodi Iterativi ----------- "

def Jacobi(A, b, x0, maxit, tol, xTrue):
  n = np.size(x0)     
  ite = 0
  x = np.copy(x0)                    
  norma_it = 1 + tol
  relErr = np.zeros((maxit, 1))       
  errIter = np.zeros((maxit, 1))      
  relErr[0] = np.linalg.norm(xTrue - x0) / np.linalg.norm(xTrue)
  while (ite < maxit - 1 and norma_it > tol):                    
    x_old=np.copy(x)                  
    for i in range(0,n):
        x[i] = (b[i] - np.dot(A[i,0:i], x_old[0:i]) - np.dot(A[i,i+1:n], x_old[i+1:n])) / A[i,i]
    ite = ite + 1                  
    norma_it = np.linalg.norm(x_old-x) / np.linalg.norm(x_old)   
    relErr[ite] = np.linalg.norm(xTrue - x) / np.linalg.norm(xTrue) 
    errIter[ite - 1] = norma_it        
  relErr = relErr[:ite]              
  errIter = errIter[:ite]          
  return [x, ite, relErr, errIter]


def GaussSeidel(A, b, x0, maxit, tol, xTrue):
    n = np.size(x0)     
    ite = 0
    x = np.copy(x0)
    norma_it = 1 + tol
    relErr = np.zeros((maxit, 1))
    errIter = np.zeros((maxit, 1))
    relErr[0] = np.linalg.norm(xTrue - x0) / np.linalg.norm(xTrue)
    while (ite < maxit - 1 and norma_it > tol):
      x_old = np.copy(x)
      for i in range(0,n):
        x[i] = (b[i] - np.dot(A[i,0:i], x[0:i]) - np.dot(A[i,i+1:n], x_old[i+1:n])) / A[i,i]
      ite = ite + 1
      norma_it = np.linalg.norm(x_old-x) / np.linalg.norm(x_old)
      relErr[ite] = np.linalg.norm(xTrue - x) / np.linalg.norm(xTrue)
      errIter[ite-1] = norma_it
    relErr = relErr[:ite]
    errIter = errIter[:ite]  
    return [x, ite, relErr, errIter]



" ----------- Esercizio 2 Test ----------- "


" Matrice tridiagonale n*n "
n = 100
A = (9*np.eye(n))+(np.diag(np.ones(n-1)*(-4),k=1))+(np.diag(np.ones(n-1)*(-4),k=-1))
xTrue = np.ones((n,1))
b = np.dot(A,xTrue)

raggio_spettrale= max(abs(np.linalg.eigvals(A)))
print('\n A:\n',A)
print("Raggio Spettrale:",raggio_spettrale,', Converge: ',raggio_spettrale < 1,'\n')

" Metodi Iterativi "
x0 = np.zeros((n,1))
x0[0]=1. 
maxit = 300
tol = 1.e-8

(xJacobi, kJacobi, relErrJacobi, errIterJacobi) = Jacobi(A,b,x0,maxit,tol,xTrue) 
(xGS, kGS, relErrGS, errIterGS) = GaussSeidel(A,b,x0,maxit,tol,xTrue ) 

print('\nSoluzione con Jacobi:' )
for i in range(n):
    print('%0.10f' %xJacobi[i])

print('\nSoluzione con Gauss-Seidel:' )
for i in range(n):
    print('%0.10f' %xGS[i])



" Grafici con confronti "

" errore relativo al variare del numero di iterazioni per dimensione fissata "
x_jacobi = range (0, kJacobi)
x_gauss = range(0, kGS)
plt.figure(figsize=(20,10))
plt.plot(x_jacobi, relErrJacobi, label='Jacobi', color='blue', linewidth=2)
plt.plot(x_gauss, relErrGS, label='Gauss-Seidel', color = 'red', linewidth=2)
plt.legend(loc='upper right')
plt.xlabel('iterazioni')
plt.ylabel('errore relativo')
plt.title('Errore relativo al variare delle iterazioni tra Jacobi e Gauss-Siedel')
plt.show()


" errore relativo finale al variare della dimensione del sistema "

dim = np.arange(2,82)
maxit = 500
tol = 1.e-8


Erel_LU = np.zeros(np.size(dim))
Erel_Cho = np.zeros(np.size(dim))
Erel_Fin_J = np.zeros(np.size(dim))
Erel_Fin_GS = np.zeros(np.size(dim))

Ite_J = np.zeros(np.size(dim))
Ite_GS = np.zeros(np.size(dim))

time_J = np.zeros(np.size(dim))
time_GS = np.zeros(np.size(dim))
time_Cho = np.zeros(np.size(dim))
time_LU = np.zeros(np.size(dim))

i=0
for n in dim:
  A = (9*np.eye(n))+(np.diag(np.ones(n-1)*(-4),k=1))+(np.diag(np.ones(n-1)*(-4),k=-1))
  xTrue = np.ones((n,1))
  b = np.dot(A,xTrue)
  x0 = np.zeros((n,1))
  x0[0]=1. 

  " Diretto - Cholesky "
  time_start = time.perf_counter()
  L = scipy.linalg.cholesky(A, True)
  y = scipy.linalg.solve(L, b)
  my_x = scipy.linalg.solve(L.T, y)
  time_end = time.perf_counter()
  time_Cho[i] = time_end - time_start
  Erel_Cho[i] = np.linalg.norm(xTrue - my_x,2) / np.linalg.norm(xTrue,2)

  " Diretto - LU "
  time_start = time.perf_counter()
  LU, P = LUdec.lu_factor(A)
  my_x = scipy.linalg.lu_solve((LU, P), b)
  time_end = time.perf_counter() 
  time_LU[i] = time_end - time_start
  Erel_LU[i] = np.linalg.norm(xTrue - my_x,2) / np.linalg.norm(xTrue,2)

  " Iterativo - Jacobi "
  time_start = time.perf_counter()
  [xJ, iteJ, ErelJ, errItJ] = Jacobi(A,b,x0,maxit,tol,xTrue)
  time_end = time.perf_counter()
  time_J[i] = time_end - time_start
  
  " Iterativo - Gauss-Seidel "
  time_start =time.perf_counter()
  [xGS, iteGS, ErelGS, errItGS] = GaussSeidel(A,b,x0,maxit,tol,xTrue)
  time_end = time.perf_counter()
  time_GS[i] = time_end - time_start

  " Errore relativo finale "
  Erel_Fin_J[i] = errItJ[iteJ-1]
  Erel_Fin_GS[i] = errItGS[iteGS-1]
  
  " Iterazioni "
  Ite_J[i] = iteJ
  Ite_GS[i] = iteGS
  
  i=i+1
  
  


" Grafico errore relativo finale al variare della dimensione del sistema"
plt.figure(figsize=(20,10))
plt.plot(dim,Erel_Fin_J , label='Jacobi', color='blue', linewidth=2)
plt.plot(dim,Erel_Fin_GS, label='Gauss-Seidel', color = 'red', linewidth=2)
plt.legend(loc='upper left')
plt.xlabel('dimensione sistema')
plt.ylabel('errore relativo finale')
plt.title('Errore relativo al variare delle dimensioni del sistema tra Jacobi e Gauss-Siedel')
plt.show() 

" Grafico numero di iterazioni al variare della dimensione del sistema"
plt.figure(figsize=(20,10))
plt.plot(dim,Ite_GS,label='Gauss-Seidel',color='blue',linewidth=2)
plt.plot(dim,Ite_J,label='Jacobi',color = 'red',linewidth=2)
plt.legend(loc='upper right')
plt.xlabel('dimensione sistema')
plt.ylabel('iterazioni')
plt.title('Iterazioni al variare della dimensione del sistema tra Jacobi e Gauss-Seidel')
plt.show()

" Grafico tempo impegato da tutti i metodi al variare della dimensione "
plt.figure(figsize=(20,10))
plt.plot(dim,time_J,label = 'Jacobi' , color = 'blue', linewidth=2)
plt.plot(dim,time_GS,label = 'Gauss-Seidel' , color = 'red', linewidth=2)
plt.plot(dim,time_Cho,label = 'Cholesky' , color = 'green', linewidth=2)
plt.plot(dim,time_LU,label = 'LU dec' , color = 'yellow', linewidth=2)
plt.legend(loc='upper right')
plt.xlabel('dimensioni sistema')
plt.ylabel('tempo')
plt.title('Tempo impiegato al variare della dimensione del sistema da tutti i metodi')
plt.show()




