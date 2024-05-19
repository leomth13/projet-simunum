#Differences finies
import numpy as np
import matplotlib.pyplot as plt

#csts et arrays: à relancer à chaque fois!
"""
Unités: distance: UA
        temps: jours terrestres
        masse: masse solaire
"""
G = 2.9591e-4
m_S = 1
m_J = 9.5450e-4
m_Sat=0.0002857

t_max=1826250
dt=30
N=int(t_max/dt)+1
t=np.linspace(0,t_max,N)

q_S=np.zeros((3,N))
p_S=np.zeros((3,N))
q_J=np.zeros((3,N))
p_J=np.zeros((3,N))

CM=np.zeros((3,N))

q_S[:,0]=np.array([0,0,0])
p_S[:,0]=np.array([0,0,0])
q_J[:,0]=np.array([2.7643030638361,4.1797287385959,-0.0792087931357])
p_J[:,0]=m_J*np.array([-0.0063872681206,0.0045222694488,0.0001241195185])
q_Sat=np.zeros((3,N))
p_Sat=np.zeros((3,N))
q_Sat[:,0]=np.array([9.1966657030107,-3.0767392639166,-0.3125144594307])
p_Sat[:,0]=m_Sat*np.array([0.0014563476466,0.0052845257699,-0.0001499232881])

#Heun 2 corps
def Heun_2(q_S,p_S,m_S,q_J,p_J,m_J,time):
    q_S_tilde=np.zeros((3,N))
    p_S_tilde=np.zeros((3,N))
    q_J_tilde=np.zeros((3,N))
    p_J_tilde=np.zeros((3,N))
    for i in range(N-1):
        q_S_tilde[:,i+1]=q_S[:,i]+dt*p_S[:,i]/m_S
        p_S_tilde[:,i+1]=p_S[:,i]-dt*(G*m_S*m_J)*(q_S[:,i]-q_J[:,i])/(np.linalg.norm(q_S[:,i]-q_J[:,i])**3)
        q_J_tilde[:,i+1]=q_J[:,i]+dt*p_J[:,i]/m_J
        p_J_tilde[:,i+1]=p_J[:,i]+dt*(G*m_S*m_J)*(q_S[:,i]-q_J[:,i])/(np.linalg.norm(q_S[:,i]-q_J[:,i])**3)

        q_S[:,i+1]=q_S[:,i]+dt/2*(p_S[:,i]/m_S+p_S_tilde[:,i+1]/m_S)
        p_S[:,i+1]=p_S[:,i]-dt/2*(G*m_S*m_J)*((q_S[:,i]-q_J[:,i])/(np.linalg.norm(q_S[:,i]-q_J[:,i])**3)+(q_S_tilde[:,i+1]-q_J_tilde[:,i+1])/(np.linalg.norm(q_S_tilde[:,i+1]-q_J_tilde[:,i+1])**3))
        q_J[:,i+1]=q_J[:,i]+dt/2*(p_J[:,i]/m_J+p_J_tilde[:,i+1]/m_J)
        p_J[:,i+1]=p_J[:,i]+dt/2*(G*m_S*m_J)*((q_S[:,i]-q_J[:,i])/(np.linalg.norm(q_S[:,i]-q_J[:,i])**3)+(q_S_tilde[:,i+1]-q_J_tilde[:,i+1])/(np.linalg.norm(q_S_tilde[:,i+1]-q_J_tilde[:,i+1])**3))

    for i in range(N):
        CM[:,i]=(m_S*q_S[:,i]+m_J*q_J[:,i])/(m_S+m_J)
    plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot(q_S[0,:]-CM[0,:], q_S[1,:]-CM[1,:], q_S[2,:]-CM[2,:],"o", label = "Soleil",color='red')
    ax.plot(q_J[0,:]-CM[0,:], q_J[1,:]-CM[1,:], q_J[2,:]-CM[2,:], color='orange', label = "Trajectoire Jupiter")
    ax.grid()
    ax.set_title("Orbite de Jupiter autour du Soleil durant 5000 ans, repère du centre de masse (Heun)")
    ax.set_xlabel("X (UA)")
    ax.set_ylabel("Y (UA)")
    ax.set_zlabel("Z (UA)")
    ax.legend()
    plt.show()
    plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot([0],[0],[0],"o", label = "Soleil",color='red')
    ax.plot(q_J[0,:]-q_S[0,:], q_J[1,:]-q_S[1,:], q_J[2,:]-q_S[2,:], label = "Trajectoire Jupiter",color='orange')
    ax.grid()
    ax.set_title("Orbite de Jupiter autour du Soleil durant 5000 ans, repère héliocentrique (Heun)")
    ax.set_xlabel("X (UA)")
    ax.set_ylabel("Y (UA)")
    ax.set_zlabel("Z (UA)")
    ax.legend()
    plt.show()
    
    #energy
    E=np.zeros(N)
    for l in range(N):
        E[l]=np.linalg.norm(p_S[:,l])**2/(2*m_S) + np.linalg.norm(p_J[:,l])**2/(2*m_J) - G*(m_S*m_J/np.linalg.norm(q_S[:,l]-q_J[:,l]))
    #angular momentum
    L=np.zeros((3,N))
    for k in range(N):
        L[:,k]=np.cross(q_S[:,k],p_S[:,k])+np.cross(q_J[:,k],p_J[:,k])
    return E,L

#Heun_2(q_S, p_S, m_S, q_J, p_J, m_J,t)
E_H2=Heun_2(q_S, p_S, m_S, q_J, p_J, m_J,t)[0]
L_H2=Heun_2(q_S, p_S, m_S, q_J, p_J, m_J,t)[1]

#Verlet 2 corps
def Verlet_2(Q_1,P_1,m_1,Q_2,P_2,m_2,time):
    for i in range (N-1):
        P_tilde_1 = np.zeros(3)
        P_tilde_2 = np.zeros(3)
        diff_1 = Q_2[:,i] - Q_1[:,i]
        dist_1 = np.linalg.norm(Q_2[:,i]-Q_1[:,i])
        
        #On fait tous les tildes
        P_tilde_1 = P_1[:,i] + (dt * G*m_1*m_2*(1/dist_1**3)*diff_1)
        P_tilde_2 = P_2[:,i] - (dt * G*m_1*m_2*(1/dist_1**3)*diff_1)
        
        #On fait tous les i+1
        Q_1[:,i+1] = Q_1[:,i] + (dt/(2*m_1) * (P_1[:,i] + P_tilde_1))
        Q_2[:,i+1] = Q_2[:,i] + (dt/(2*m_2) * (P_2[:,i] + P_tilde_2))
        
        diff_2 = Q_2[:,i+1] - Q_1[:,i+1]
        dist_2 = np.linalg.norm(Q_2[:,i+1]-Q_1[:,i+1])
        
        P_1[:,i+1] = P_1[:,i] + (dt/2 * (G*m_1*m_2*(1/dist_1**3)*diff_1 + G*m_1*m_2*(1/dist_2**3)*diff_2))
        P_2[:,i+1] = P_2[:,i] - (dt/2 * (G*m_1*m_2*(1/dist_1**3)*diff_1 + G*m_1*m_2*(1/dist_2**3)*diff_2))
    for i in range(N):
        CM[:,i]= (m_1*Q_1[:,i]+m_2*Q_2[:,i])/(m_1+m_2)
    
   #plots
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot3D(Q_1[0,:]-CM[0,:],Q_1[1,:]-CM[1,:],Q_1[2,:]-CM[2,:],"o",color='red', label = 'Soleil')
    ax.plot3D(Q_2[0,:]-CM[0,:],Q_2[1,:]-CM[1,:],Q_2[2,:]-CM[2,:],color='orange', label = 'Jupiter')
    plt.legend()
    plt.title("Trajectoire de Jupiter autour du Soleil durant 5000 ans, repère du centre de masse (Stormer-Verlet)")
    ax.set_xlabel("X (UA)")
    ax.set_ylabel("Y (UA)")
    ax.set_zlabel("Z (UA)")
    plt.show()
    
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter([0],[0],[0],"o", label = 'Soleil', color= 'red')
    ax.plot3D(Q_2[0,:]-Q_1[0,:],Q_2[1,:]-Q_1[1,:],Q_2[2,:]-Q_1[2,:], label='Jupiter',color='orange')
    plt.legend()
    plt.title('Trajectoire de Jupiter autour du Soleil durant 5000 ans, repère héliocentrique (Stormer-Verlet)')
    ax.set_xlabel("X (UA)")
    ax.set_ylabel("Y (UA)")
    ax.set_zlabel("Z (UA)")
    
    #energy
    E=np.zeros(N)
    for l in range(N):
        E[l]=np.linalg.norm(P_1[:,l])**2/(2*m_1) + np.linalg.norm(P_2[:,l])**2/(2*m_2) - G*(m_1*m_2/np.linalg.norm(Q_1[:,l]-Q_2[:,l]))
    #angular momentum
    L=np.zeros((3,N))
    for k in range(N):
        L[:,k]=np.cross(Q_1[:,k],P_1[:,k])+np.cross(Q_2[:,k],P_2[:,k])
    return E,L

#Verlet_2(q_S,p_S,m_S,q_J,p_J,m_J,t)
E_V2=Verlet_2(q_S,p_S,m_S,q_J,p_J,m_J,t)[0]
L_V2=Verlet_2(q_S,p_S,m_S,q_J,p_J,m_J,t)[1]

#Heun3
def Heun3(q_S,p_S,m_S,q_J,p_J,m_J,q_A,p_A,m_A,time):
    q_S_tilde=np.zeros((3,N))
    p_S_tilde=np.zeros((3,N))
    q_J_tilde=np.zeros((3,N))
    p_J_tilde=np.zeros((3,N))
    q_A_tilde=np.zeros((3,N))
    p_A_tilde=np.zeros((3,N))
    for i in range(N-1):
        q_S_tilde[:,i+1]=q_S[:,i]+dt*p_S[:,i]/m_S
        p_S_tilde[:,i+1]=p_S[:,i]-dt*G*m_S*(m_J*(q_S[:,i]-q_J[:,i])/(np.linalg.norm(q_S[:,i]-q_J[:,i])**3)+m_A*(q_S[:,i]-q_A[:,i])/(np.linalg.norm(q_S[:,i]-q_A[:,i])**3))
        q_J_tilde[:,i+1]=q_J[:,i]+dt*p_J[:,i]/m_J
        p_J_tilde[:,i+1]=p_J[:,i]+dt*G*m_J*(m_S*(q_S[:,i]-q_J[:,i])/(np.linalg.norm(q_S[:,i]-q_J[:,i])**3)-m_A*(q_J[:,i]-q_A[:,i])/(np.linalg.norm(q_J[:,i]-q_A[:,i])**3))
        q_A_tilde[:,i+1]=q_A[:,i]+dt*p_A[:,i]/m_A
        p_A_tilde[:,i+1]=p_A[:,i]+dt*G*m_A*(m_S*(q_S[:,i]-q_A[:,i])/(np.linalg.norm(q_S[:,i]-q_A[:,i])**3)+m_J*(q_J[:,i]-q_A[:,i])/(np.linalg.norm(q_J[:,i]-q_A[:,i])**3))

        q_S[:,i+1]=q_S[:,i]+dt/2*(p_S[:,i]/m_S+p_S_tilde[:,i+1]/m_S)
        p_S[:,i+1]=p_S[:,i]-dt/2*G*m_S*(m_J*((q_S[:,i]-q_J[:,i])/(np.linalg.norm(q_S[:,i]-q_J[:,i])**3)+(q_S_tilde[:,i+1]-q_J_tilde[:,i+1])/(np.linalg.norm(q_S_tilde[:,i+1]-q_J_tilde[:,i+1])**3))+m_A*((q_S[:,i]-q_A[:,i])/(np.linalg.norm(q_S[:,i]-q_A[:,i])**3)+(q_S_tilde[:,i+1]-q_A_tilde[:,i+1])/(np.linalg.norm(q_S_tilde[:,i+1]-q_A_tilde[:,i+1])**3)))
        q_J[:,i+1]=q_J[:,i]+dt/2*(p_J[:,i]/m_J+p_J_tilde[:,i+1]/m_J)
        p_J[:,i+1]=p_J[:,i]+dt/2*G*m_J*(m_S*((q_S[:,i]-q_J[:,i])/(np.linalg.norm(q_S[:,i]-q_J[:,i])**3)+(q_S_tilde[:,i+1]-q_J_tilde[:,i+1])/(np.linalg.norm(q_S_tilde[:,i+1]-q_J_tilde[:,i+1])**3))-m_A*((q_J[:,i]-q_A[:,i])/(np.linalg.norm(q_J[:,i]-q_A[:,i])**3)+(q_J_tilde[:,i+1]-q_A_tilde[:,i+1])/(np.linalg.norm(q_J_tilde[:,i+1]-q_A_tilde[:,i+1])**3)))
        q_A[:,i+1]=q_A[:,i]+dt/2*(p_A[:,i]/m_A+p_A_tilde[:,i+1]/m_A)
        p_A[:,i+1]=p_A[:,i]+dt/2*G*m_A*(m_S*((q_S[:,i]-q_A[:,i])/(np.linalg.norm(q_S[:,i]-q_A[:,i])**3)+(q_S_tilde[:,i+1]-q_A_tilde[:,i+1])/(np.linalg.norm(q_S_tilde[:,i+1]-q_A_tilde[:,i+1])**3))+m_J*((q_J[:,i]-q_A[:,i])/(np.linalg.norm(q_J[:,i]-q_A[:,i])**3)+(q_J_tilde[:,i+1]-q_A_tilde[:,i+1])/(np.linalg.norm(q_J_tilde[:,i+1]-q_A_tilde[:,i+1])**3)))

    for i in range(N):
        CM[:,i]=(m_S*q_S[:,i]+m_J*q_J[:,i]+m_A*q_A[:,i])/(m_S+m_J+m_A)
    #CM
    plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot(q_S[0,:]-CM[0,:], q_S[1,:]-CM[1,:], q_S[2,:]-CM[2,:],"o", color='red', label = "Soleil")
    ax.plot(q_J[0,:]-CM[0,:], q_J[1,:]-CM[1,:], q_J[2,:]-CM[2,:],color='orange', label = "Trajectoire Jupiter")
    ax.plot(q_A[0,:]-CM[0,:], q_A[1,:]-CM[1,:], q_A[2,:]-CM[2,:],color='blue', label = "Trajectoire Saturne")
    ax.grid()
    ax.set_title("Orbites de Jupiter et de Saturne autour du Soleil durant 5000 ans, repère du centre de masse (Heun)")
    ax.set_xlabel("X (UA)")
    ax.set_ylabel("Y (UA)")
    ax.set_zlabel("Z (UA)")
    ax.legend()
    plt.show()
    #HC
    plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot([0],[0],[0], "o",color='red', label = "Soleil")
    ax.plot(q_J[0,:]-q_S[0,:], q_J[1,:]-q_S[1,:], q_J[2,:]-q_S[2,:],color='orange', label = "Trajectoire Jupiter")
    ax.plot(q_A[0,:]-q_S[0,:], q_A[1,:]-q_S[1,:], q_A[2,:]-q_S[2,:],color='blue', label = "Trajectoire Saturne")
    ax.grid()
    ax.set_title("Orbites de Jupiter et de Saturne autour du Soleil durant 5000 ans, repère héliocentrique (Heun)")
    ax.set_xlabel("X (UA)")
    ax.set_ylabel("Y (UA)")
    ax.set_zlabel("Z (UA)")
    ax.legend()
    plt.show()
    #Invariants
    E=np.zeros(N)
    L=np.zeros((3,N))
    for k in range(N):
        L[:,k]=np.cross(q_S[:,k], p_S[:,k])+np.cross(q_J[:,k], p_J[:,k])+np.cross(q_A[:,k], p_A[:,k])
        E[k]=np.linalg.norm(p_S[:,k])**2/(2*m_S)+np.linalg.norm(p_J[:,k])**2/(2*m_J)+np.linalg.norm(p_A[:,k])**2/(2*m_A)-G*(m_S*m_J/np.linalg.norm(q_S[:,k]-q_J[:,k])+m_S*m_A/np.linalg.norm(q_S[:,k]-q_A[:,k])+m_J*m_A/np.linalg.norm(q_J[:,k]-q_A[:,k]))
    """
    plt.figure()
    plt.plot(time, E, label= "Energie en fonction du temps")
    plt.title("Conservation de l'energie")
    plt.xlabel("Temps")
    plt.ylabel("Energie")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(time, L[0,:], label= "L_x")
    plt.plot(time, L[1,:], label= "L_y")
    plt.plot(time, L[2,:], label= "L_z")
    plt.title("Conservation du moment angulaire")
    plt.xlabel("Temps")
    plt.ylabel("Moment angulaire")
    plt.legend()
    plt.show()
    """
    return E,L

#Heun3(q_S, p_S, m_S, q_J, p_J, m_J, q_Sat, p_Sat, m_Sat,t)
E_H3=Heun3(q_S, p_S, m_S, q_J, p_J, m_J, q_Sat, p_Sat, m_Sat,t)[0]
L_H3=Heun3(q_S, p_S, m_S, q_J, p_J, m_J, q_Sat, p_Sat, m_Sat,t)[1]

#Verlet 3 corps
def Verlet3(Q_1,P_1,m_1,Q_2,P_2,m_2,Q_3,P_3,m_3,time):
    for i in range(N-1):
        P_tilde_1 = np.zeros(3)
        P_tilde_2 = np.zeros(3)
        P_tilde_3 = np.zeros(3)
        diff_21_i = Q_2[:, i] - Q_1[:, i]
        diff_12_i = Q_1[:, i] - Q_2[:, i]
        diff_31_i = Q_3[:, i] - Q_1[:, i]
        diff_13_i = Q_1[:, i] - Q_3[:, i]
        diff_23_i = Q_2[:, i] - Q_3[:, i]
        diff_32_i = Q_3[:, i] - Q_2[:, i]
        dist_21_i = np.linalg.norm(Q_2[:, i] - Q_1[:, i])
        dist_12_i = np.linalg.norm(Q_1[:, i] - Q_2[:, i])
        dist_31_i = np.linalg.norm(Q_3[:, i] - Q_1[:, i])
        dist_13_i = np.linalg.norm(Q_1[:, i] - Q_3[:, i])
        dist_23_i = np.linalg.norm(Q_2[:, i] - Q_3[:, i])
        dist_32_i = np.linalg.norm(Q_3[:, i] - Q_2[:, i])
        
        # Calculate all the tildes
        P_tilde_1 = P_1[:, i] + (dt * (G * m_1 * m_2 * (1 / dist_21_i**3) * diff_21_i + G * m_1 * m_3 * (1 / dist_31_i**3) * diff_31_i))
        P_tilde_2 = P_2[:, i] + (dt * (G * m_1 * m_2 * (1 / dist_12_i**3) * diff_12_i + G * m_2 * m_3 * (1 / dist_32_i**3) * diff_32_i))
        P_tilde_3 = P_3[:, i] + (dt * (G * m_1 * m_3 * (1 / dist_13_i**3) * diff_13_i + G * m_2 * m_3 * (1 / dist_23_i**3) * diff_23_i))
        
        # Calculate all the i+1 values
        Q_1[:, i+1] = Q_1[:, i] + (dt / (2 * m_1) * (P_1[:, i] + P_tilde_1))
        Q_2[:, i+1] = Q_2[:, i] + (dt / (2 * m_2) * (P_2[:, i] + P_tilde_2))
        Q_3[:, i+1] = Q_3[:, i] + (dt / (2 * m_3) * (P_3[:, i] + P_tilde_3))
        
        diff_21_f = Q_2[:, i+1] - Q_1[:, i+1]
        diff_12_f = Q_1[:, i+1] - Q_2[:, i+1]
        diff_31_f = Q_3[:, i+1] - Q_1[:, i+1]
        diff_13_f = Q_1[:, i+1] - Q_3[:, i+1]
        diff_23_f = Q_2[:, i+1] - Q_3[:, i+1]
        diff_32_f = Q_3[:, i+1] - Q_2[:, i+1]
        dist_21_f = np.linalg.norm(Q_2[:, i+1] - Q_1[:, i+1])
        dist_12_f = np.linalg.norm(Q_1[:, i+1] - Q_2[:, i+1])
        dist_31_f = np.linalg.norm(Q_3[:, i+1] - Q_1[:, i+1])
        dist_13_f = np.linalg.norm(Q_1[:, i+1] - Q_3[:, i+1])
        dist_23_f = np.linalg.norm(Q_2[:, i+1] - Q_3[:, i+1])
        dist_32_f = np.linalg.norm(Q_3[:, i+1] - Q_2[:, i+1])
        
        P_1[:, i+1] = P_1[:, i] + (dt/2 * ((G*m_1*m_2*(1/dist_21_i**3)*diff_21_i + G*m_1*m_3*(1/dist_31_i**3)*diff_31_i) + (G*m_1*m_2*(1/dist_21_f**3)*diff_21_f + G*m_1*m_3*(1/dist_31_f**3)*diff_31_f)))
        P_2[:, i+1] = P_2[:, i] + (dt/2 * ((G*m_1*m_2*(1/dist_12_i**3)*diff_12_i + G*m_2*m_3*(1/dist_32_i**3)*diff_32_i) + (G*m_1*m_2*(1/dist_12_f**3)*diff_12_f + G*m_2*m_3*(1/dist_32_f**3)*diff_32_f)))
        P_3[:, i+1] = P_3[:, i] + (dt/2 * ((G*m_1*m_3*(1/dist_13_i**3)*diff_13_i + G*m_2*m_3*(1/dist_23_i**3)*diff_23_i) + (G*m_1*m_3*(1/dist_13_f**3)*diff_13_f + G*m_2*m_3*(1/dist_23_f**3)*diff_23_f)))

    for i in range(N):
        CM[:,i]=(m_1*Q_1[:,i]+m_2*Q_2[:,i]+m_3*Q_3[:,i])/(m_1+m_2+m_3)
    #graphes
    
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot3D(Q_1[0,:]-CM[0,:],Q_1[1,:]-CM[1,:],Q_1[2,:]-CM[2,:],"o", label = 'Soleil',color='red')
    ax.plot3D(Q_2[0,:]-CM[0,:],Q_2[1,:]-CM[1,:],Q_2[2,:]-CM[2,:], label = 'Jupiter',color='orange')
    ax.plot3D(Q_3[0,:]-CM[0,:],Q_3[1,:]-CM[1,:],Q_3[2,:]-CM[2,:], label = 'Saturne',color='blue')
    plt.legend()
    plt.title("Orbites de Jupiter et de Saturne autour du Soleil durant 5000 ans, repère du centre de masse (Stormer-Verlet)")
    ax.set_xlabel("X (UA)")
    ax.set_ylabel("Y (UA)")
    ax.set_zlabel("Z (UA)")
    plt.show()
    
    plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter([0],[0],[0],"o", label = 'Soleil', color= 'red')
    ax.plot3D(Q_2[0,:]-Q_1[0,:],Q_2[1,:]-Q_1[1,:],Q_2[2,:]-Q_1[2,:], label='Jupiter',color='orange')
    ax.plot3D(Q_3[0,:]-Q_1[0,:],Q_3[1,:]-Q_1[1,:],Q_3[2,:]-Q_1[2,:], label='Saturne',color='blue')
    plt.legend()
    plt.title('Orbites de Jupiter et de Saturne autour du Soleil durant 5000 ans, repère héliocentrique (Stormer-Verlet)')
    ax.set_xlabel("X (UA)")
    ax.set_ylabel("Y (UA)")
    ax.set_zlabel("Z (UA)")
    
    #energy
    E=np.zeros(N)
    for l in range(N):
        E[l]=np.linalg.norm(P_1[:,l])**2/(2*m_1) + np.linalg.norm(P_2[:,l])**2/(2*m_2) + np.linalg.norm(P_3[:,l])**2/(2*m_3) - G*((m_1*m_3/np.linalg.norm(Q_1[:,l]-Q_3[:,l])) + (m_1*m_2/np.linalg.norm(Q_1[:,l]-Q_2[:,l])) + (m_2*m_3/np.linalg.norm(Q_2[:,l]-Q_3[:,l])))
    #angular momentum
    L=np.zeros((3,N))
    for k in range(N):
        L[:,k]=np.cross(Q_1[:,k],P_1[:,k])+np.cross(Q_2[:,k],P_2[:,k])+np.cross(Q_3[:,k],P_3[:,k])
    return E,L

#Verlet3(q_S,p_S,m_S,q_J,p_J,m_J,q_Sat,p_Sat,m_Sat,t)
E_V3=Verlet3(q_S,p_S,m_S,q_J,p_J,m_J,q_Sat,p_Sat,m_Sat,t)[0]
L_V3=Verlet3(q_S,p_S,m_S,q_J,p_J,m_J,q_Sat,p_Sat,m_Sat,t)[1]

#Plots combinés des invariants 2 corps
plt.figure()
plt.plot(t,E_H2,color='red',label='Energie du système (Heun)')
plt.plot(t,E_V2,color='orange',label='Energie du système (Stormer-Verlet)')
plt.xlabel('temps')
plt.ylabel('Energie')
plt.title("Conservation de l'énergie")
plt.legend()
plt.show()
plt.figure()
plt.plot(t,L_H2[0,:],label='coordonnée x (Heun)',color='blue')
plt.plot(t,L_V2[0,:],label='coordonnée x (Stormer-Verlet)',color='cyan')
plt.xlabel('temps')
plt.ylabel('Moment angulaire')
plt.legend()
plt.title('Conservation du moment angulaire (x)')
plt.show()
plt.figure()
plt.plot(t,L_H2[1,:],label='coordonnée y (Heun)',color='lime')
plt.plot(t,L_V2[1,:],label='coordonnée y (Stormer-Verlet)',color='green')
plt.xlabel('temps')
plt.ylabel('Moment angulaire')
plt.legend()
plt.title('Conservation du moment angulaire (y)')
plt.show()
plt.figure()
plt.plot(t,L_H2[2,:],label='coordonnée z (Heun)',color='red')
plt.plot(t,L_V2[2,:],label='coordonnée z (Stormer-Verlet)',color='tomato')
plt.xlabel('temps')
plt.ylabel('Moment angulaire')
plt.legend()
plt.title('Conservation du moment angulaire (z)')
plt.show()

#Plots combinés des invariants 3 corps
plt.figure()
plt.plot(t,E_H3,color='red',label='Energie du système (Heun)')
plt.plot(t,E_V3,color='orange',label='Energie du système (Stormer-Verlet)')
plt.xlabel('temps')
plt.ylabel('Energie')
plt.title("Conservation de l'énergie")
plt.legend()
plt.show()
plt.figure()
plt.plot(t,L_H3[0,:],label='coordonnée x (Heun)',color='blue')
plt.plot(t,L_V3[0,:],label='coordonnée x (Stormer-Verlet)',color='cyan')
plt.xlabel('temps')
plt.ylabel('Moment angulaire')
plt.legend()
plt.title('Conservation du moment angulaire (x)')
plt.show()
plt.figure()
plt.plot(t,L_H3[1,:],label='coordonnée y (Heun)',color='lime')
plt.plot(t,L_V3[1,:],label='coordonnée y (Stormer-Verlet)',color='green')
plt.xlabel('temps')
plt.ylabel('Moment angulaire')
plt.legend()
plt.title('Conservation du moment angulaire (y)')
plt.show()
plt.figure()
plt.plot(t,L_H3[2,:],label='coordonnée z (Heun)',color='red')
plt.plot(t,L_V3[2,:],label='coordonnée z (Stormer-Verlet)',color='tomato')
plt.xlabel('temps')
plt.ylabel('Moment angulaire')
plt.legend()
plt.title('Conservation du moment angulaire (z)')
plt.show()
"""
plt.figure()
plt.xlim(0,t_max/365.25)
plt.ylim(-2.7125e-8,-2.711e-8)
plt.xticks([1000,2000,3000,4000,5000])
plt.plot(t,E_H2)
"""