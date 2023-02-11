import numpy as np
import scipy
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
import matplotlib.pyplot as plt


#(a) matrice generica
# n=np.random.randint(low)
n = 4
np.random.rand(n, n)
# A1 matrice di numeri casuali
A = abs(np.random.randint(10, 1000, (n, n)))
print('A:\n', A)

x = np.ones((n, 1))
b = np.dot(A, x)

#(b) numero di condizione
condA = np.linalg.cond(A)
print('K(A)=', condA, '\n')

#(c) fattorizzazione L U con pivoting
#matrice quadrata e det. diverso da 0
#Ly=b | Ux=y

LU, P = LUdec.lu_factor(A)
print('LU: \n', LU)
print('P: \n', P, '\n')

my_x = scipy.linalg.lu_solve((LU, P), b)
print('my_x = \n', my_x, '\n')


#(a) matrice di Hilbert
n1 = np.random.randint(2, 15)
H = scipy.linalg.hilbert(n1)
x = np.ones((n1, 1))
b = np.dot(H, x)

#(b) numero di condizione
condH = np.linalg.cond(H)
print('\n', 'Matrice di Hilbert:', '\n', H)
print('K(H)=', condH, '\n')

#(c) fattorizzazione di Cholesky
#Hilber è mal condizionata
L = scipy.linalg.cholesky(H, True)
print('errore = ', scipy.linalg.norm(H - np.matmul(L, np.transpose(L)), 'fro'))

y = scipy.linalg.solve(L, b)
my_x = scipy.linalg.solve(L.T, y)
print('my_x = \n ', my_x, '\n')

#(a) A con diagonale tutti 9 e -4 sopra e sotto diagonali
A = 9 * np.eye(n) - 4 * np.eye(n, k=-1) - 4 * np.eye(n, k=1)
A = A @ A.T
x = np.ones((n, 1))
b = np.dot(A, x)
print('\n', 'A tridiag. simmetrica def. positiva: \n', A)

#(b) numero di condizione
condA = np.linalg.cond(A)
print('K(A)=', condA, '\n')

#(c) fattorizzazione di Cholesky
L = scipy.linalg.cholesky(A, True)
print('errore = ', scipy.linalg.norm(A-np.matmul(L, np.transpose(L)), 'fro'))

y = scipy.linalg.solve(L, b)
my_x = scipy.linalg.solve(L.T, y)
print('my_x = \n', my_x)


def Plot(t):

    if t == "Hilbert":
        range = 14
    else:
        range = 100
    K_A = np.zeros((range-2, 1))
    Err = np.zeros((range-2, 1))

    for n in np.arange(2, range):
        if (t == "Hilbert"):
            A = scipy.linalg.hilbert(n)
            text = "Hilbert (Cholesky)"
        elif(t == "Generica"):
            #np.random.rand(n, n)
            A = abs(np.random.randint(10, 1000, (n, n)))
            text = "Generica (Dec. LU)"
        else:
            A = 9 * np.eye(n) - 4 * np.eye(n, k=-1) - 4 * np.eye(n, k=1)
            A = A@A.T
            text = "Matrice tridiagonale (Cholesky)"
        x = np.ones((n, 1))
        b = np.dot(A, x)

        if(t == "Generica"):
            # Decomposizione LU con pivoting
            LU, P = LUdec.lu_factor(A)
            my_x = scipy.linalg.lu_solve((LU, P), b)  
        else:
            # Cholesky
            L = scipy.linalg.cholesky(A, True)
            y = scipy.linalg.solve(L, b)
            my_x = scipy.linalg.solve(L.T, y)

        # numero di condizione
        K_A[n-2] = np.linalg.cond(A)
        # errore relativo
        Err[n-2] = np.linalg.norm(x - my_x, 2) / np.linalg.norm(x, 2)

    title = "Numero di Condizione " + str(text)
    plt.figure(figsize=(20,10))
    plt.subplot(1, 2, 1)
    x_plot = np.arange(2, range)
    plt.plot(x_plot, K_A, color='blue', linewidth=1)
    plt.title(title)
    plt.xlabel('dim')
    plt.ylabel('K(A)')

    title = "Errore relativo " + str(text)
    plt.subplot(1, 2, 2)
    plt.plot(x_plot, Err, color='red', linewidth=1)
    plt.title(title)
    plt.xlabel('dim')
    plt.ylabel('Err')
    plt.show()

Plot("Generica")
Plot("Hilbert")
Plot("triangpos")
