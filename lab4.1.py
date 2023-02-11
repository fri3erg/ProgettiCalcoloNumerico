import numpy as np
import math
import matplotlib.pyplot as plt
import math
import time

" ----------- Esercizio 1 Zero di una funzione ----------- "


" Metodo di Bisezione "
def bisezione(a, b, f, tolx, xTrue):
  #k numero minimo di iterazioni per tolleranza tolx
  k = math.ceil(np.log2(abs(b - a) / tolx))
  vecErrore = np.zeros((k,1)) 
 
  if f(a)*f(b)>0:
      print("Errore: non c'è uno 0 per la funzione f")
      return ()
      
  for i in range(1,k):
      c = a + (b - a) / 2
      vecErrore[i-1] = math.fabs(xTrue - c)

      if abs(f(c)) < 1.e-6: #f(c) è molto vicino a 0
          return (c, i, k, vecErrore)
      else:
        if np.sign(f(c)) > 0: 
          b = c
        else:
          a = c

  return (c, i, k, vecErrore)


" Metodo di Newton "
def newton( f, df, tolf, tolx, maxit, xTrue, x0=0):
    err = np.zeros(maxit, dtype = float)
    vecErrore = np.zeros( (maxit,1), dtype = float)
    i = 0
    err[0] = tolx + 1
    vecErrore[0] = np.abs(x0 - xTrue)
    x = x0
    xprec = x0
       
    while (i <= maxit and (math.fabs(f(xprec)) >= tolf and math.fabs(f(x - xprec)) >= tolx)): 
        x = xprec - f(xprec) / df(xprec)
        err[i] = math.fabs((x - xprec))
        vecErrore[i] = math.fabs((x - xTrue))
        xprec = x
        i = i + 1 
       
    err = err[0:i]
    vecErrore = vecErrore[0:i]
     
    return (x, i, err, vecErrore)  


" Creazione del problema test"
f = lambda x: np.exp(x) - (x ** 2)
df = lambda x: math.exp(x) - 2*x
xTrue = -0.7034674
fTrue = f(xTrue)
print (" fTrue: ",fTrue,'\n',"Xtrue: ",xTrue,"\n")

a=-1.0
b=1.0
tolx= 10**(-6)
tolf = 10**(-6)
maxit=100
x0 = 0


" Soluzione con il metodo di Bisezione "
sol, ib, k, vecErroreb = bisezione(a,b,f,tolx,xTrue)
print('Soluzione con il metodo di Bisezione \n x =', f(sol),'\n iter_bise=',ib , '\n iter_max=', k,'\n')

" Soluzione con il metodo Newton "
sol2, inw, err, vecErroren = newton(f, df, tolf, tolx, maxit, xTrue) 
print('Soluzione con il metodo di Newton \n x =',sol2,'\n iter_new=', inw, '\n err_new=', err,'\n')

" Grafico funzione in [a,b]"
x_plot = np.linspace(a, b, 101)
y_plot = f(x_plot)

plt.figure(figsize=(25,10))
plt.plot(x_plot,y_plot, color = 'blue')
plt.plot(xTrue,0,"o",color = 'red')
plt.title("Funzione")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.axvline(x=0, c="black", label="x=0") #ascisse
plt.axhline(y=0, c="black", label="y=0") #ordinata
plt.grid()
plt.show()

" Grafico Errore al variare delle iterazioni"

plt.title("Errore al variare delle iterazioni")
plt.plot(np.arange(0, ib), vecErroreb[0:ib],color = 'red', label="Bisezione")
plt.plot(np.arange(0, inw), err[0:inw], color = 'blue', label="Newton")
plt.legend()
plt.xlabel('iterazioni')
plt.ylabel('errore')
plt.grid()
plt.show()


" ----------- Esercizio 2 Approssimazioni successive ----------- "

def approssimazioni_successive(f, g, tolf, tolx, x0, xTrue, maxit):
    x = x0
    xprec = x0
    i = 0
    vecErrore = np.zeros( maxit + 1, dtype = float)
    while(math.fabs(f(xprec)) >= tolf and math.fabs(f(x - xprec)) >= tolx and i <= maxit):
        x = g(xprec)
        xprec = x
        vecErrore[i] = math.fabs((x - xTrue))
        i = i + 1 
    return (x, i, vecErrore)

" Creazione del problema test "
g1 = lambda x : x - f(x)*math.exp(x/2)
g2 = lambda x : x - f(x)*math.exp(-x/2)
g3 = lambda x : x - f(x)/df(x)
xTrue = -0.7034674
fTrue = f(xTrue)

a=-1.0
b=1.0
tolx= 10**(-6)
tolf = 10**(-6)
maxit = 100

#g1
x_g1, it_g1, vecErrore_g1 = approssimazioni_successive(f,g1,tolf,tolx,0, xTrue, maxit)
print('g1 - Punto Fisso \n x =', x_g1,'\n iter=',it_g1 , '\n')
#g2
x_g2, it_g2, vecErrore_g2 = approssimazioni_successive(f,g2,tolf,tolx,0, xTrue,maxit)
print('g2 - Punto Fisso \n x =', x_g2,'\n iter=',it_g2 , '\n')
#g3
x_g3, it_g3, vecErrore_g3 = approssimazioni_successive(f,g3,tolf,tolx,0, xTrue, maxit)
print('g3 - Punto Fisso \n x =', x_g3,'\n iter=',it_g3 , '\n')


plt.plot(np.arange(0, it_g1), vecErrore_g1[0:it_g1],color = 'green')
plt.title("Errore Punto Fisso al variare delle iterazioni di g1(x)")
plt.grid()
plt.xlabel('iterazioni')
plt.ylabel('errore punto fisso')
plt.show()

plt.plot(np.arange(0, it_g2), vecErrore_g2[0:it_g2],color = 'red')
plt.title("Errore Punto Fisso al variare delle iterazioni di g2(x)")
plt.grid()
plt.xlabel('iterazioni')
plt.ylabel('errore punto fisso')
plt.show()

plt.plot(np.arange(0, it_g3), vecErrore_g3[0:it_g3],color = 'blue')
plt.title("Errore Punto Fisso al variare delle iterazioni di iter g3(x)")
plt.grid()
plt.xlabel('iterazioni')
plt.ylabel('errore punto fisso')
plt.show()

plt.plot(np.arange(0,it_g1), vecErrore_g1[0:it_g1], color='green', label="g1")
plt.plot(np.arange(0,it_g2), vecErrore_g2[0:it_g2], color='red',label="g2")
plt.plot(np.arange(0,it_g3), vecErrore_g3[0:it_g3], color='blue',label="g3")
plt.legend(loc='upper right')
plt.xlabel('iterazioni')
plt.ylabel('errore punto fisso')
plt.title("Confronto tra g1, g2, g3 al variare delle iterazioni")
plt.grid()
plt.show()



" ----------- Esercizio 3 Confrontare e commentare le prestazioni dei tre metodi ----------- "


" Creazione del problema"
def confronto_metodi(f,df,g,a,b,xTrue,x0,title,funz):
    sTime = np.zeros(3)
    start = time.perf_counter()
    bis = bisezione(a,b,f,tolx,xTrue)
    end = time.perf_counter()
    sTime[0] = end - start
    
    start = time.perf_counter()
    new = newton(f, df, tolf, tolx, 100, xTrue, x0)
    end = time.perf_counter()
    sTime[1] = end - start
    
    start = time.perf_counter()
    app = approssimazioni_successive(f, g, tolf, tolx, x0, xTrue, 100)
    end = time.perf_counter()
    sTime[2] = end - start
    
    print(title)
    #numero di iterazioni
    print("Numero di iterazioni \nBisezione: ",bis[1],"\nNewton: ",new[1],"\nApp. Successive: ",app[1],"\n")
    #tempi di esecuzione
    print("Tempi di esecuzioni \nBisezione: ",sTime[0],"\nNewton: ",sTime[1],"\nApp. Successive: ",sTime[2],"\n")
    #errore
    print("Errore commesso \nBisezione: ",(bis[3])[bis[1]-1],"\nNewton: ",(new[3])[new[1]-1],"\nApp. Successive: ",(app[2])[app[1]-1],"\n")
    
    if(funz==1):
        x1_plot=np.linspace(0,2,101)
    
    else:
        x1_plot=np.linspace(3,5,101)
    
    f1_plot = np.zeros(x1_plot.size)
    for index, value in enumerate(x1_plot):
        if (funz == 1):
            f1_plot[index] = f1(value)
        else:
            f1_plot[index] = f2(value)

    axis = np.zeros((101, 1))


    " Grafico funzione "
    plt.plot(x1_plot, f1_plot, linewidth='2', color='blue', label=title)
    plt.grid()
    plt.plot(xTrue, 0, "o", color='red', label='punto F(x)=0')
    plt.title('Grafico di f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    " Grafico Errore al variare delle iterazioni "

    plt.title("Errore tra tutti i metodi al variare delle iterazioni")
    plt.plot(range(bis[1]), bis[3][0:bis[1]], color='green', label="bisezione")
    plt.plot(range(new[1]), new[3][0:new[1]], color='blue', label="newton")
    plt.plot(range(app[1]), app[2][0:app[1]], color='red', label="app.succ.")
    plt.xlabel('iterazioni')
    plt.ylabel('errore')
    plt.grid()
    plt.legend()
    plt.show()


f1 = lambda x: x**3 + 4*x*math.cos(x) - 2
df1 = lambda x: x * 3*x- 4 * math.sin(x) + 4 * math.cos(x)
g1 = lambda x: (2-x**3) / (4*math.cos(x))
xTrue1 = 0.536839
a1 = 0
b1 = 2
title =str("f(x) = x^3 + 4xcos(x) - 2\n")
confronto_metodi(f1,df1,g1,a1,b1,xTrue1,0,title, 1)

f2 = lambda x: x - x**(1/3) - 2
df2= lambda x: 1 - (1/ (3*pow(x,2/3)))
g2 = lambda x: x**(1/3) + 2
xTrue2 = 3.5214
a2 = 3
b2 = 5
title = str("f(x) = x - x^(1/3) - 2\n")
confronto_metodi(f2,df2,g2,a2,b2,xTrue2,3,title, 2)



