#importing important libs

import numpy as np
import scipy as sci
import matplotlib as mlt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


#Non-Dimensionalisation

#Universal Gravitation Constant
G=6.67408e-11 #N-m2/kg2

#Reference quantities
m_nd=1.989e+30 #kg #mass of the sun
r_nd=5.326e+12 #m #distance between stars in Alpha Centauri
v_nd=30000 #m/s #relative velocity of earth around the sun
t_nd=79.91*365.25*24*3600 #s #orbital period of Alpha Centauri

#Net constants
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd


def com_pos(M1,R1,M2,R2):
    return (M1*R1 + M2*R2)/(M1+M2)

def com_vel(M1,V1,M2,V2):
    return (M1*V1 + M2*V2)/(M1+M2)


#Defining masses 
m1 = 1.1 # Aplha Centauri A
m2 = 0.907 # Aplha Centauri B

#Defining initial position and velocity vectors
r1 = [-0.5,0,0] #m
r2 = [0.5,0,0] #m
v1 = [0.01,0.01,0] #m/s
v2 = [-0.05,0,-0.1] #m/s

#Converting pos and vel vectors to arrays
r1 = np.array(r1) 
r2 = np.array(r2) 
v1 = np.array(v1)
v2 = np.array(v2)

#Find pos and vel COM
r_com = com_pos(m1,r1,m2,r2)
v_com = com_vel(m1,r1,m2,r2)



#Defining func for the equations of motion
def Two_Body_eqn(w,t,G,m1,m2):
    r1 = w[:3]
    r2 = w[3:6]
    v1 = w[6:9]
    v2 = w[9:12]
    
    r=sci.linalg.norm(r2-r1) #Calculating norm of vector
    
    #eqn. 3.1
    a_1 = K1*m2*(r2-r1) / r**3 # a_1 is d(v1)/dt
    a_2 = K1*m1*(r1-r2) / r**3
    #eqn 3.2
    v_1 = K2*v1 # v_1 is d(r1)/dt
    v_2 = K2*v2
    
    r_derivs = np.concatenate((v_1, v_2))
    derivs = np.concatenate((r_derivs, a_1 ,a_2))
    return derivs



#package initial parameters
init_params = np.array([r1,r2,v1,v2]).flatten()
time_span = np.linspace(0,15,1500) # 5 orbital periods and 500 points

# Running the ODE solver
import scipy.integrate as si

Two_Body_sol = si.odeint(Two_Body_eqn,init_params,time_span,args=(G,m1,m2))

r1_sol = Two_Body_sol[:,:3]
r2_sol = Two_Body_sol[:,3:6]

# Make the figure
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111,projection='3d') 

#Create new arrays for animation, this gives you the flexibility
#to reduce the number of points in the animation if it becomes slow
#Currently set to select every 4th point
r1_anim = r1_sol[::1,:].copy()
r2_anim = r2_sol[::1,:].copy()

#Set initial marker for planets, that is, blue and red circles at the initial positions
h1 = [ax.scatter(r1_anim[0,0],r1_anim[0,1],r1_anim[0,2],color="darkblue",marker="o",s=80,label="Alpha Centauri A")]
h2 = [ax.scatter(r2_anim[0,0],r2_anim[0,1],r2_anim[0,2],color="tab:red",marker="o",s=80,label="Alpha Centauri B")]

#Add a few bells and whistles
ax.set_xlabel("x",fontsize=14)
ax.set_ylabel("y",fontsize=14)
ax.set_zlabel("z",fontsize=14)
ax.set_title("Visualization of orbits of stars in a 2-body system\n",fontsize=16)
ax.legend(loc="upper left",fontsize=14)


# Creating a function Animate that changes plots every frame('i' - frame no.)
def Animate_2b(i,head1,head2):
    #Remove old markers
    h1[0].remove()
    h2[0].remove()
    
    # Plotting the orbits (for every i, we plot from init pos to final pos)
    t1 = ax.plot(r1_anim[:i,0],r1_anim[:i,1],r1_anim[:i,2],color='darkblue')
    t2 = ax.plot(r2_anim[:i,0],r2_anim[:i,1],r2_anim[:i,2],color='r')
    
    # Plotting the current markers
    h1[0]=ax.scatter(r1_anim[i,0],r1_anim[i,1],r1_anim[i,2],color="darkblue",marker="o",s=80)
    h2[0]=ax.scatter(r2_anim[i,0],r2_anim[i,1],r2_anim[i,2],color="r",marker="o",s=80)
    return t1,t2,h1,h2


# Using the function module to make the animation
#If used in Jupyter Notebook, animation will not display only a static image will display with this command
# anim_2b = animation.FuncAnimation(fig,Animate_2b,frames=1000,interval=5,repeat=False,blit=False,fargs=(h1,h2))

anim_2b = animation.FuncAnimation(fig,Animate_2b,frames=1000,interval=2,repeat=False,blit=False,fargs=(h1,h2))


# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=4000)

#To save animation to disk, enable this command
anim_2b.save("TwoBodyProblem_test2.mp4", writer=writer, dpi=300)


