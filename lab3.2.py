import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from skimage import data

" ----------- Esercizio 3 Approssimazione funzioni tramite Minini quadrati ----------- "

# gradi del polinomio
n = [1,2,3,5,7]
# punti noti
m = 20

def p(alpha, x):
        A = np.zeros((x.size,len(alpha)))
        for i in range(len(alpha)):
          A[:,i] = x**i
        y =  A @ alpha
        return y


def alpha_svd_(f,a,b,i):
    x = np.linspace(a,b,m)
    y = f(x)
    
    A = np.zeros((m, i+1))
    for k in range(i+1):
        A[:,k] = x**k
    U, S, V  = scipy.linalg.svd(A)
    alpha_svd = np.zeros(S.shape)
    for j in range(i+1):
        uj = U[:,j] 
        vj = V[j,:]
        alpha_svd = alpha_svd + ((uj @ y) * vj ) / S[j]
        
    return alpha_svd
                
          
def plot(f1,f2,f3):
    
    for i in n:
        
        a1=alpha_svd_(f1,-1,1,i)
        a2=alpha_svd_(f2,-1,1,i)
        a3=alpha_svd_(f3,1,5,i)
        
        x1=np.linspace(-1, 1, m)
        y1=f1(x1)
        x1_plot = np.linspace(-1,1,100)
        y1_plot = f1(x1_plot)
        y1_svd = p(a1,x1)
        y1_svd_plot = p(a1,x1_plot)
        

        x2=np.linspace(-1, 1, m)
        y2=f2(x2)
        x2_plot = np.linspace(-1,1,100)
        y2_plot = f2(x2_plot)
        y2_svd = p(a2,x2)
        y2_svd_plot = p(a2,x2_plot)
        
                
        x3=np.linspace(1, 5, m)
        y3=f3(x3)
        x3_plot = np.linspace(1,5,100)
        y3_plot = f3(x3_plot)
        y3_svd = p(a3,x3)
        y3_svd_plot = p(a3,x3_plot)
        
        if(i!=2):#non polinomio grado 2
            err1 = np.linalg.norm(y1_svd - f1(x1))
            print("errore commesso sui nodi di f1 al grado ", i, " : ",err1)
          
            err2 = np.linalg.norm(y2_svd - f2(x2))
            print("errore commesso sui nodi di f2 al grado ", i, " : ",err2)
            
            err3 = np.linalg.norm(y3_svd - f3(x3))
            print("errore commesso sui nodi di f3 al grado ", i, " : ",err3,"\n")
        
        print("errore commesso in x=0 in f1 al grado ", i, " : ",abs(p(a1,np.array([0])) - f1(0)))
        print("errore commesso in x=0 in f2 al grado ", i, " : ",abs(p(a2,np.array([0])) - f2(0)))
        print("errore commesso in x=0 in f3 al grado ", i, " : ",abs(p(a3,np.array([0])) - f3(0)), "\n")
        
        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.plot(x1_plot,y1_plot,color = "red",label = "f1")
        plt.plot(x1,y1,"o",color = "red")
        plt.plot(x1_plot,y1_svd_plot,color = "red",label = "p(x) grado n = " + str(i))
        plt.legend()
        plt.title("Approssimazione tramite SVD della f1 con n=%i"%i)
        
        plt.subplot(1,3,2)
        plt.plot(x2_plot,y2_plot,color = "green",label = "f2 ")
        plt.plot(x2,y2,"o",color = "green")
        plt.plot(x2_plot,y2_svd_plot,color = "green",label = "p(x) grado n = " + str(i))
        plt.legend()
        plt.title("Approssimazione tramite SVD della f2 con n=%i"%i)
        
        plt.subplot(1,3,3)
        plt.plot(x3_plot,y3_plot,color = "blue",label = "f3 ")
        plt.plot(x3,y3,"o",color = "blue")
        plt.plot(x3_plot,y3_svd_plot,color = "blue",label = "p(x) grado n = " + str(i))
        plt.legend()
        plt.title("Approssimazione tramite SVD della f3 con n=%i"%i)
        
        plt.show()
    

f1 = lambda x: x * np.exp(x) 
f2 = lambda x: 1/(1 + 25 * x)
f3 = lambda x: np.sin(5 * x) + 3 * x

plot(f1,f2,f3)


" ----------- Esercizio 4 Compressione immagini tramite SVD ----------- "


immagine_1 = data.chelsea()
A=immagine_1[:,:,2]

print(type(A))
print(A.shape)


plt.imshow(A, cmap='gray')
plt.show()

U, S, Vt = scipy.linalg.svd(A)
print (S)

" Calcolo diadi "

A_p = np.zeros(A.shape)
p_max = 10

for i in range(p_max): #?
    Uj=U[:,i]
    Vj=Vt[i,:]
    tmp=np.outer(Uj,Vj)
    A_p =A_p + S[i] *np.outer(Uj,Vj)
    
" Errore relativo e fattore di compressione "
err_rel = scipy.linalg.norm(A-A_p,2)/scipy.linalg.norm(A,2)
c = (1/p_max)* min(np.shape(A)[0],np.shape(A)[1]) - 1

print('\n')
print('L\'errore relativo della ricostruzione di A è', err_rel)
print('Il fattore di compressione è: ', c)

plt.figure(figsize=(20, 10))

fig1 = plt.subplot(1, 2, 1)
fig1.imshow(A, cmap='gray')
plt.title('Immagine originale')

fig2 = plt.subplot(1, 2, 2)
fig2.imshow(A_p, cmap='gray')
plt.title('Immagine compressa con grado del polinomio p =' + str(p_max))

plt.show()

err_rel = np.zeros(p_max)
c = np.zeros(p_max)

" errore relativo e fattore di compressione al variare di p "
for p in range(1,p_max+1):
    A_p = np.zeros(A.shape)
    for i in range(p):
        Uj=U[:,i]
        Vj=Vt[i,:]
        tmp=np.outer(Uj,Vj)
        A_p =A_p + S[i] *np.outer(Uj,Vj)
        
    err_rel[p-1] = scipy.linalg.norm(A-A_p)/scipy.linalg.norm(A)

    c[p-1] = (1/p)* min(np.shape(A)[0],np.shape(A)[1]) - 1   


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.plot (np.arange(1,p_max+1),c,color='red',linewidth=1)
plt.title('fattore di compressione al variare del numero di diadi')
plt.legend(['fattore di compressione'])

plt.subplot(1,2,2)
plt.plot (np.arange(1,p_max+1),err_rel,color='blue',linewidth=1)
plt.title('Errore relativo al variare del numero di diadi')
plt.legend('errore relativo')

plt.show()
    

" ----------- Esercizio 5 Compressione immagini tramite SVD ----------- "


immagine = [plt.imread('windows.jpeg').astype(np.float64)/255.0 , plt.imread('luna.jpg').astype(np.float64)/255.0]

for immagine1 in immagine:
    A=immagine1[:,:,2]


    print(type(A))
    print(A.shape)
    
    plt.figure(figsize=(7,10))
    plt.imshow(A, cmap='gray')
    plt.show()
    
    U, S, Vt = scipy.linalg.svd(A)
    print (S)
    
    " Calcolo diadi "
    
    A_p = np.zeros(A.shape)
    p_max = 10
    
    for i in range(p_max):
        Uj=U[:,i]
        Vj=Vt[i,:]
        tmp=np.outer(Uj,Vj)
        A_p =A_p + S[i] *np.outer(Uj,Vj)
        
    " Errore relativo e fattore di compressione "
    err_rel = scipy.linalg.norm(A-A_p,2)/scipy.linalg.norm(A,2)
    c = (1/p_max)* min(np.shape(A)[0],np.shape(A)[1]) - 1
    
    print('\n')
    print('L\'errore relativo della ricostruzione di A è', err_rel)
    print('Il fattore di compressione è: ', c)
    
    plt.figure(figsize=(20, 10))
    
    fig1 = plt.subplot(1, 2, 1)
    fig1.imshow(A, cmap='gray')
    plt.title('Immagine originale')
    
    fig2 = plt.subplot(1, 2, 2)
    fig2.imshow(A_p, cmap='gray')
    plt.title('Immagine compressa con grado del polinomio p =' + str(p_max))
    
    plt.show()
    
    err_rel = np.zeros(p_max)
    c = np.zeros(p_max)
    
    " errore relativo e fattore di compressione al variare di p "
    for p in range(1,p_max+1):
        A_p = np.zeros(A.shape)
        for i in range(p):
            Uj=U[:,i]
            Vj=Vt[i,:]
            tmp=np.outer(Uj,Vj)
            A_p =A_p + S[i] *np.outer(Uj,Vj)
            
        err_rel[p-1] = scipy.linalg.norm(A-A_p)/scipy.linalg.norm(A)
    
        c[p-1] = (1/p)* min(np.shape(A)[0],np.shape(A)[1]) - 1   
    
    
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.plot (np.arange(1,p_max+1),c,color='red',linewidth=1)
    plt.title('fattore di compressione al variare del numero di diadi')
    plt.legend(['fattore di compressione'])
    
    plt.subplot(1,2,2)
    plt.plot (np.arange(1,p_max+1),err_rel,color='blue',linewidth=1)
    plt.title('Errore relativo al variare del numero di diadi')
    plt.legend('errore relativo')
    
    plt.show()
    