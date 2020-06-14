#Importing important libraries

import scipy as sci
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Non-Dimensionalisation

G=6.67408e-11 #N-m2/kg2

#Reference quantities
m_nd=1.989e+30 #kg
r_nd=5.326e+12 #m
v_nd=30000 #m/s
t_nd=79.91*365.25*24*3600 #s

#Net constants
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd

#Define masses
m1=1.1 #Star 1
m2=0.907 #Star 2
m3=1.425 #Star 3


#Define initial position vectors
r1=[-0.5,1,0] #m
r2=[0.5,0,0.5] #m
r3=[0.2,1,1.5] #m

#Convert pos vectors to arrays
r1=np.array(r1)
r2=np.array(r2)
r3=np.array(r3)

#Find Centre of Mass
r_com=(m1*r1+m2*r2+m3*r3)/(m1+m2+m3)

#Define initial velocities
v1=[0.02,0.02,0.02] #m/s
v2=[-0.05,0,-0.1] #m/s
v3=[0,-0.03,0]

#Convert velocity vectors to arrays
v1=np.array(v1)
v2=np.array(v2)
v3=np.array(v3)

#Find velocity of COM
v_com=(m1*v1+m2*v2+m3*v3)/(m1+m2+m3)


def ThreeBodyEquations(w,t,G,m1,m2):
    #Unpack all the variables from the array "w"
    r1=w[:3]
    r2=w[3:6]
    r3=w[6:9]
    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18]
    
    #Find out distances between the three bodies
    r12=sci.linalg.norm(r2-r1)
    r13=sci.linalg.norm(r3-r1)
    r23=sci.linalg.norm(r3-r2)
    
    #Define the derivatives according to the equations
    dv1bydt=K1*m2*(r2-r1)/r12**3+K1*m3*(r3-r1)/r13**3
    dv2bydt=K1*m1*(r1-r2)/r12**3+K1*m3*(r3-r2)/r23**3
    dv3bydt=K1*m1*(r1-r3)/r13**3+K1*m2*(r2-r3)/r23**3
    dr1bydt=K2*v1
    dr2bydt=K2*v2
    dr3bydt=K2*v3
    
    #Package the derivatives into one final size-18 array
    r12_derivs=np.concatenate((dr1bydt,dr2bydt))
    r_derivs=np.concatenate((r12_derivs,dr3bydt))
    v12_derivs=np.concatenate((dv1bydt,dv2bydt))
    v_derivs=np.concatenate((v12_derivs,dv3bydt))
    derivs=np.concatenate((r_derivs,v_derivs))
    return derivs


#Package initial parameters
init_params=np.array([r1,r2,r3,v1,v2,v3]) #Package initial parameters into one size-18 array
init_params=init_params.flatten() #Flatten the array to make it 1D
time_span=np.linspace(0,20,1000) #Time span is 20 orbital years and 1000 points


#Run the ODE solver
import scipy.integrate
three_body_sol=sci.integrate.odeint(ThreeBodyEquations,init_params,time_span,args=(G,m1,m2))


#Store the position solutions into three distinct arrays
r1_sol=three_body_sol[:,:3]
r2_sol=three_body_sol[:,3:6]
r3_sol=three_body_sol[:,6:9]


#Plot the orbits of the three bodies
fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(111,projection="3d")
ax.plot(r1_sol[:,0],r1_sol[:,1],r1_sol[:,2],color="mediumblue")
ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="red")
ax.plot(r3_sol[:,0],r3_sol[:,1],r3_sol[:,2],color="gold")
ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="darkblue",marker="o",s=80,label="Star 1")
ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="darkred",marker="o",s=80,label="Star 2")
ax.scatter(r3_sol[-1,0],r3_sol[-1,1],r3_sol[-1,2],color="goldenrod",marker="o",s=80,label="Star 3")
ax.set_xlabel("x-coordinate",fontsize=14)
ax.set_ylabel("y-coordinate",fontsize=14)
ax.set_zlabel("z-coordinate",fontsize=14)
ax.set_title("Visualization of orbits of stars in a 3-body system\n",fontsize=14)
ax.legend(loc="upper left",fontsize=14)


#Animate the orbits of the three bodies


#Make the figure 
fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(111,projection="3d")

#Create new arrays for animation, this gives you the flexibility
#to reduce the number of points in the animation if it becomes slow
#Currently set to select every 4th point
r1_sol_anim=r1_sol[::1,:].copy()
r2_sol_anim=r2_sol[::1,:].copy()
r3_sol_anim=r3_sol[::1,:].copy()

#Set initial marker for planets, that is, blue,red and green circles at the initial positions
head1=[ax.scatter(r1_sol_anim[0,0],r1_sol_anim[0,1],r1_sol_anim[0,2],color="darkblue",marker="o",s=80,label="Star 1")]
head2=[ax.scatter(r2_sol_anim[0,0],r2_sol_anim[0,1],r2_sol_anim[0,2],color="darkred",marker="o",s=80,label="Star 2")]
head3=[ax.scatter(r3_sol_anim[0,0],r3_sol_anim[0,1],r3_sol_anim[0,2],color="goldenrod",marker="o",s=80,label="Star 3")]

#Create a function Animate that changes plots every frame (here "i" is the frame number)
def Animate(i,head1,head2,head3):
    #Remove old markers
    head1[0].remove()
    head2[0].remove()
    head3[0].remove()
    
    #Plot the orbits (every iteration we plot from initial position to the current position)
    trace1=ax.plot(r1_sol_anim[:i,0],r1_sol_anim[:i,1],r1_sol_anim[:i,2],color="mediumblue")
    trace2=ax.plot(r2_sol_anim[:i,0],r2_sol_anim[:i,1],r2_sol_anim[:i,2],color="red")
    trace3=ax.plot(r3_sol_anim[:i,0],r3_sol_anim[:i,1],r3_sol_anim[:i,2],color="gold")
    
    #Plot the current markers
    head1[0]=ax.scatter(r1_sol_anim[i-1,0],r1_sol_anim[i-1,1],r1_sol_anim[i-1,2],color="darkblue",marker="o",s=100)
    head2[0]=ax.scatter(r2_sol_anim[i-1,0],r2_sol_anim[i-1,1],r2_sol_anim[i-1,2],color="darkred",marker="o",s=100)
    head3[0]=ax.scatter(r3_sol_anim[i-1,0],r3_sol_anim[i-1,1],r3_sol_anim[i-1,2],color="goldenrod",marker="o",s=100)
    return trace1,trace2,trace3,head1,head2,head3,

#Some beautifying
ax.set_xlabel("x-coordinate",fontsize=14)
ax.set_ylabel("y-coordinate",fontsize=14)
ax.set_zlabel("z-coordinate",fontsize=14)
ax.set_title("Visualization of orbits of stars in a 3-body system\n",fontsize=14)
ax.legend(loc="upper left",fontsize=14)


#If used in Jupyter Notebook, animation will not display only a static image will display with this command
# anim_2b = animation.FuncAnimation(fig,Animate_2b,frames=1000,interval=5,repeat=False,blit=False,fargs=(h1,h2))


#Use the FuncAnimation module to make the animation
repeatanim=animation.FuncAnimation(fig,Animate,frames=800,interval=10,repeat=False,blit=False,fargs=(head1,head2,head3))

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=4000)

#To save animation to disk, enable this command
repeatanim.save("ThreeBodyProblem.mp4", writer=writer)