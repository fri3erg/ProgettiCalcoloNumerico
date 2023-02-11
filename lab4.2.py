import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


" -------- Esercizio 4 Minimizzazione utilizzando il metodo del gradiente -------- "

def alpha_backtracking(x,grad): # algoritmo di backtracking per la scelta della lunghezza del passo
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
  
  x=np.zeros((2,MAXITERATION))
  norm_grad_list=np.zeros((1,MAXITERATION)) 
  function_eval_list=np.zeros((1,MAXITERATION))
  error_list=np.zeros((1,MAXITERATION)) 
  
  k=0
  x_last = np.array([x0[0],x0[1]])
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
  

  print('iterations =',k)
  print('last guess: x=(%f,%f)'%(x[0,k],x[1,k]))
  print ('f(xTrue)= ',f(x_true))

  return (x_last,norm_grad_list, function_eval_list, error_list, k, x[:,:k])



"Creazione del problema test"

x_true=np.array([1,2])

def f(x):       
    return 10*pow(x[0]-1,2)+ pow(x[1]-2,2)

def grad_f(x):
    return np.array([20*(x[0]-1),2*(x[1]-2)])


step=0.1
maxIterations=1000
ABSOLUTE_STOP=1.e-5
x0 = np.array((3,-5))


[x_last,norm_grad_list, function_eval_list, error_list, k, puntixy] = my_minimize(x0,x_true,step,maxIterations,ABSOLUTE_STOP)


numero_punti = puntixy.shape[1]

x = np.linspace(0.25,3.0,numero_punti)
y = np.linspace(-5,2,numero_punti)
X, Y = np.meshgrid(x, y)
Z=f([X,Y])

plt.figure(figsize=(10, 5))

"superficie"
ax1 = plt.subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Grafico superficie f(x)')
ax1.view_init(elev=20)

ax2 = plt.subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis')
ax2.set_title("Grafico superficie da un'altra angolazione")
ax2.view_init(elev=5)
plt.show()

" Grafico curve di livello"
contours = plt.contour(X, Y, Z, levels=50)
plt.plot(puntixy[0],puntixy[1])
plt.title('Curve di Livello')
plt.show()

" Grafico confronti "
iterations_plot=range(k+1)

"Norma Gradidente al variare delle iterazioni"
plt.figure(figsize=(20, 5))
plt.subplot(1,3,1)
plt.plot(iterations_plot,norm_grad_list[0,:],linewidth=1)
plt.xlabel('iterazioni')
plt.grid()
plt.ylabel('valore norma del gradiente')
plt.title('Norma Gradidente al variare delle iterazioni')

"Errore al variare delle iterazioni"
plt.subplot(1,3,2)
plt.plot(iterations_plot,error_list[0,:],linewidth=1)
plt.xlabel('iterazioni')
plt.ylabel('errore')
plt.grid()
plt.title('Errore al variare delle iterazioni')

"Funzione Obiettivo al variare delle iterazioni"
plt.subplot(1,3,3)
plt.plot(iterations_plot,function_eval_list[0,:],linewidth=1)
plt.xlabel('iterazioni')
plt.ylabel('valore funzione')
plt.grid()
plt.title('Funzione Obiettivo al variare delle iterazioni')
plt.show()









