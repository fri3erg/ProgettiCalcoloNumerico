import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

" -------- Esercizio 5 Minimizzazione utilizzando il metodo del gradiente con step variabile -------- "

def alpha_backtracking(x,grad): # algoritmo di backtracking pper la scelta della lunghezza del passo
  alpha = 1.1
  rho = 0.5
  c1 = 0.25
  p = -1*grad
  j=0
  jmax=10
  alphamin= 10**-7
  while (f(x + (alpha * p)) > f(x) + c1 * alpha * grad.T @ p) and alpha>alphamin and (j<jmax): #condizione di Armijio
      alpha=alpha*rho
      j=j+1
      
  if j >= jmax or alpha <= alphamin:
      return -1
      
  return alpha
    
def my_minimize(x0,x_true,step,MAXITERATION,ABSOLUTE_STOP): 
  
  x=np.zeros((N,MAXITERATION))
  norm_grad_list=np.zeros((1,MAXITERATION)) 
  function_eval_list=np.zeros((1,MAXITERATION))
  error_list=np.zeros((1,MAXITERATION)) 
  
  k=0
  x_last = np.array(x0)
  x[:,k] = x_last
  function_eval_list[:,k]= abs (f(x_last))
  error_list[:,k]= np.linalg.norm(x_last-x_true,2)
  norm_grad_list[:,k] = np.linalg.norm(grad_f(x_last),2)
 
  while (np.linalg.norm(grad_f(x_last))>ABSOLUTE_STOP and k < MAXITERATION ):
      
      k=k+1
      grad = grad_f(x_last)
      
      # backtracking step
      step = alpha_backtracking(x_last, grad)
    
      if(step==-1):
          return -1

      x_last= x[:,k-1] - step * grad
      x[:,k] = x_last
      function_eval_list[:,k]= abs(f(x_last))
      error_list[:,k]= np.linalg.norm(x_last-x_true,2)
      norm_grad_list[:,k]= np.linalg.norm(grad_f(x_last),2)
      
  
  function_eval_list = function_eval_list[:,:k+1]
  error_list = error_list [:,:k+1]
  norm_grad_list = norm_grad_list[:,:k+1]
  print("\nLambda = ",Lambda)
  print('iterations =',k)
  print('last guess: x=(%f,%f)'%(x[0,k],x[1,k]))

  return (x_last,norm_grad_list, function_eval_list, error_list, k)



" Creazione del problema test "

def f(x): 
    b=np.ones(N)
    return pow(np.linalg.norm(x-b,2),2) + Lambda * pow(np.linalg.norm(x,2),2)

def grad_f(x): #gradiente di f
    grad=np.ones(N)
    for i in range(0,N):
        grad[i]=2*((Lambda+1)*x[i]-1)
    return grad    


N=2     #dimensione
step=0.1
x0 = np.zeros(N)
MAXITERATIONS=1000
ABSOLUTE_STOP=1.e-5


" Grafici "



#primo caso:
Lambda=0.1
xTrue= np.ones(N)*(1/(Lambda + 1))
[x_last,norm_grad_list, function_eval_list, error_list, k] = my_minimize(x0,xTrue,step,MAXITERATIONS,ABSOLUTE_STOP)

iterations_plot=range(k+1)

plt.title('lambda = 0.1')
plt.plot(iterations_plot,function_eval_list[0,:],color='green',label="funzione obiettivo", linewidth=2)
plt.plot(iterations_plot,error_list[0,:],color='blue',label="errore", linewidth=2)
plt.plot(iterations_plot,norm_grad_list[0,:],color='red',label="norma del gradiente", linewidth=2)
plt.legend()
plt.grid()
plt.show()

#secondo caso
Lambda=0.5
xTrue= np.ones(N)*(1/(Lambda + 1))
[x_last,norm_grad_list, function_eval_list, error_list, k]=my_minimize(x0,xTrue,step,MAXITERATIONS,ABSOLUTE_STOP)

iterations_plot=range(k+1)

plt.title('lambda = 0.5')

plt.plot(iterations_plot,function_eval_list[0,:],color='green',label="funzione obiettivo", linewidth=2)
plt.plot(iterations_plot,error_list[0,:],color='blue',label="errore", linewidth=2)
plt.plot(iterations_plot,norm_grad_list[0,:],color='red',label="norma del gradiente", linewidth=2)
plt.legend()
plt.grid()
plt.show()

#terzo caso
Lambda=0.7
xTrue= np.ones(N)*(1/(Lambda + 1))
[x_last,norm_grad_list, function_eval_list, error_list, k]=my_minimize(x0,xTrue,step,MAXITERATIONS,ABSOLUTE_STOP)

iterations_plot=range(k+1)

plt.title('lambda = 0.7')
plt.plot(iterations_plot,function_eval_list[0,:],color='green',label="funzione obiettivo", linewidth=2)
plt.plot(iterations_plot,error_list[0,:],color='blue',label="errore",linewidth=2)
plt.plot(iterations_plot,norm_grad_list[0,:],color='red',label="norma del gradiente", linewidth=2)
plt.legend()
plt.grid()
plt.show()

#quarto caso
Lambda=1
xTrue= np.ones(N)*(1/(Lambda + 1))
[x_last,norm_grad_list, function_eval_list, error_list, k]=my_minimize(x0,xTrue,step,MAXITERATIONS,ABSOLUTE_STOP)

iterations_plot=range(k+1)

plt.title('lambda = 1')

plt.plot(iterations_plot,function_eval_list[0,:],color='green',label="funzione obiettivo", linewidth=2)
plt.plot(iterations_plot,error_list[0,:],color='blue',label="errore", linewidth=2)
plt.plot(iterations_plot,norm_grad_list[0,:],color='red',label="norma del gradiente", linewidth=2)
plt.legend()
plt.grid()
plt.show()

