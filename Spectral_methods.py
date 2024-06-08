import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy as sc
import scipy.constants as cst


def spectral_u(N = 1024, L = 100, tmax = 200, dt = 0.05, nu = 1, MakePlot = False):
    dx = L / N
    k = np.arange(-N/2, N/2, 1)
    x = np.arange(0, L, dx)
    M = int(tmax / dt) + 1
    t = np.linspace(0, tmax, M)
    f_Lk = (2 * cst.pi * k / L) ** 2 - nu * (2 * cst.pi * k / L) ** 4
    
    #u_k array contenant les vecteurs fft
    #u array solution
    u_k = np.zeros((N,M), dtype = "complex")
    u = np.zeros((N,M))
    
    #condition initiale
    u0 = np.cos(2 * cst.pi * x / L) + 0.1 * np.cos(4 * cst.pi * x / L)
    u[:,0] = u0
    u[:,1]= u0
    u_k[:,0] = sc.fft.fftshift(sc.fft.fft(u0))
    u_k[:,1]= sc.fft.fftshift(sc.fft.fft(u0))
    
    #Crank-Nicolson/Adams Bashforth
    for i in range(1, M-1):
        u_k2 = sc.fft.fftshift(sc.fft.fft(u[:,i] ** 2)) #Transformee de Fourier de u(x,t)^2
        u_k12 = sc.fft.fftshift(sc.fft.fft(u[:,i-1] ** 2)) #Transformee de Fourier de u(x,t-dt)^2
        u_k[:,i+1] = (1 + dt / 2 * f_Lk) / (1 - dt / 2 * f_Lk) * u_k[:,i] - 1j * cst.pi / L * k * dt * (3/2 * u_k2 - 1/2 * u_k12) / (1 - dt / 2 * f_Lk)
        u[:,i+1] = np.real(sc.fft.ifft(sc.fft.ifftshift(u_k[:,i+1])))
    
    #plot
    if MakePlot==True:
        plt.figure()
        xx,tt = np.meshgrid(x, t)
        contU = plt.contourf(xx, tt, u.T, np.linspace(np.min(np.min(u)), np.max(np.max(u)), 100), cmap = cm.jet)
        plt.colorbar(contU, location="right", orientation = "vertical")
        plt.suptitle(f"Kuramoto-Sivashinsky - L={L} - ν={nu} ")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.show()
    return u, N, nu, M

#répliquer la figure de la question 1
SPL1 = spectral_u(1024, 100, 200, 0.05, nu = 1, MakePlot = True)
#Différents plots de u pour des valeurs de L
spectral_u(1024, 40, 200, 0.05, nu = 1, MakePlot = True)
spectral_u(1024, 30, 200, 0.05, nu = 1, MakePlot = True)
spectral_u(1024, 20, 200, 0.05, nu = 1, MakePlot = True)
spectral_u(1024, 10, 200, 0.05, nu = 1, MakePlot = True)
#L critique
spectral_u(1024, 6, 200, 0.05, nu = 1, MakePlot = True)
spectral_u(1024, 6.2, 200, 0.05, nu = 1, MakePlot = True)
spectral_u(1024, 6.25, 200, 0.05, nu = 1, MakePlot = True)
spectral_u(1024, 6.27, 200, 0.05, nu = 1, MakePlot = True)
spectral_u(1024, 6.3, 200, 0.05, nu = 1, MakePlot = True)
#grand temps
spectral_u(1024, 100, 1000, 0.05, nu = 1, MakePlot = True)
spectral_u(1024, 23, 500, 0.05, nu = 19, MakePlot = True)

#Amplitude
L1 = np.arange(1, 50, 0.25)
def ampl(list_L, nu, t_max, MakePlot = True):
    A, AA = [],[]
    for l in list_L:
        U, N, nu, M = spectral_u(N = 1024, L = l, nu = nu, tmax = t_max, MakePlot = False)
        A.append(np.sqrt(sum(U[:,-1] ** 2)/N))
    for j in range(1, len(A)):
        if A[j]>1e-3:
            AA.append(j)
            
    if MakePlot==True:
        plt.figure()
        plt.plot(list_L, A)
        plt.xlabel('Valeurs de L')
        plt.ylabel('Valeurs de A')
        plt.title(f'Amplitude A en fonction de L - ν={nu} - tmax={t_max}')
    
    return list_L[min(AA)]

ampl(L1, 1, 500)
#changement de nu
spectral_u(1024, 100, 200, 0.05, nu = 2, MakePlot = True)
spectral_u(1024, 100, 200, 0.05, nu = 100, MakePlot  = True)
#L_nu fct de nu
L_nu = np.arange(1, 10, 1)
def L_c(list_L, list_nu):
    L_crit=[]
    for i in list_nu:
        L_crit.append(ampl(list_L, i, 500, MakePlot = False))
    plt.figure()
    fit = sc.optimize.curve_fit(lambda nu, p, c1, c2: c1 + c2 * (nu ** p), list_nu, L_crit)
    p = fit[0][0]
    c1 = fit[0][1]
    c2 = fit[0][2]
    plt.plot(list_nu, L_crit, "^")
    plt.plot(list_nu, c1 + c2 * (list_nu ** p), color = "red", label = "Fit donnees")
    plt.legend()
    plt.title('L_ν en fonction de ν')
    plt.xlabel('ν')
    plt.ylabel('L_ν')
    plt.show()
    return p

p = L_c(L1, L_nu)
