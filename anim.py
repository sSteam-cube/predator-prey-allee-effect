import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm  # Import tqdm for progress bar



#%%

#parameters
dt=0.01
dx=0.2
l=100
t_final=1000
gamma=5
beta=0.34
delta=0.5996
epsilon=1



x=np.arange(0,l+dx,dx)
t=np.arange(0,t_final+dt,dt)
boundaryconditionu=[0,0]
boundaryconditionv=[0,0]
n=len(x)
m=len(t)

#tqdm
progress_bar1 = tqdm(total=n*m, desc="Processing")


def initial(u0,delta,l_value,dl):
  temp = int(l_value // dl) + 1
  out = np.zeros(temp)
  point1 = l_value / 4
  point2 = 3 * l_value / 4
  for j in range(temp):
    i = j * dl
    if abs(point1 - i) <= delta / 2 or abs(point2 - i) <= delta / 2:
      out[j] = u0
  return out

def fu(u, v):
  return gamma*u*(u-beta)*(1-u) - u*v



def fv(u, v):
  return u*v - delta*v



initialconditionu=initial(1,19,l+dx,dx)
initialconditionv=initial(0.1,4,l+dx,dx)



def rungestep(lu,lv,dt,dx):
  l1=lu.copy()
  l2=lv.copy()
  for i in range(1,n-1):
    k1u = dt*((lu[i-1] + lu[i+1] - 2*lu[i])/dx**2 + fu(lu[i],lv[i]))
    k1v = dt*(epsilon*(lv[i-1] + lv[i+1] - 2*lv[i])/dx**2 + fv(lu[i],lv[i]))

    k2u = dt*((lu[i-1] + lu[i+1] - 2*(lu[i] + 0.5*k1u))/dx**2 + fu(lu[i]+0.5*k1u, lv[i]+0.5*k1v))
    k2v = dt*(epsilon*(lv[i-1] + lv[i+1] - 2*(lv[i] + 0.5*k1v))/dx**2 + fv(lu[i]+0.5*k1u, lv[i]+0.5*k1v))

    k3u = dt*((lu[i-1] + lu[i+1] - 2*(lu[i] + 0.5*k2u))/dx**2 + fu(lu[i]+0.5*k2u, lv[i]+0.5*k2v))
    k3v = dt*(epsilon*(lv[i-1] + lv[i+1] - 2*(lv[i] + 0.5*k2v))/dx**2 + fv(lu[i]+0.5*k2u, lv[i]+0.5*k2v))

    k4u = dt*((lu[i-1] + lu[i+1] - 2*(lu[i] + k3u))/dx**2 + fu(lu[i]+k3u, lv[i]+k3v))
    k4v = dt*(epsilon*(lv[i-1] + lv[i+1] - 2*(lv[i] + k3v))/dx**2 + fv(lu[i]+k3u, lv[i]+k3v))

    l1[i] += (k1u + 2*k2u + 2*k3u + k4u)/6
    l2[i] += (k1v + 2*k2v + 2*k3v + k4v)/6
    progress_bar1.update(1)
  return l1,l2



U=np.zeros((m,n))
V=np.zeros((m,n))
U[:, 0] = boundaryconditionu[0]
U[:, -1] = boundaryconditionu[1]
V[:, 0] = boundaryconditionv[0]
V[:, -1] = boundaryconditionv[1]

U[0,:]=initialconditionu
V[0,:]=initialconditionv

for i in range(1,m):
  U[i,:], V[i,:] = rungestep(U[i-1,:], V[i-1,:], dt, dx)
  U[i, 0] = boundaryconditionu[0]
  U[i, -1] = boundaryconditionu[1]
  V[i, 0] = boundaryconditionv[0]
  V[i, -1] = boundaryconditionv[1]
progress_bar1.close()

#%%

def plot(time, array1,array2):
  t_index = int(time // dt)
  plt.plot(x, array1[t_index, :], label='prey')
  plt.plot(x,array2[t_index, :], linestyle='--', label='predator')
  plt.xlabel('Position')
  plt.ylabel('Population Density')
  plt.title(f'population distribution at time={time}, prey influx={boundaryconditionu[0]}, predator influx={boundaryconditionv[0]} ')
  plt.grid(True)
  plt.legend()
  plt.show()
# progress_bar.close()

def total_population(u,v,time,dt,dx):
  t_index = int(time // dt)
  usum=sum(u[t_index,:])
  vsum=sum(v[t_index,:])
  return dx*usum,dx*vsum

def phasespace(u,v,t_final,dt,dx):
  uout=[]
  vout=[]
  tin=0
  while tin<=t_final:
    utemp,vtemp = total_population(u,v,tin,dt,dx)
    uout.append(utemp)
    vout.append(vtemp)
    tin+=dt
  plt.plot(uout[len(vout)//4::],vout[len(vout)//4::])  #[len(vout)//2::]
  plt.title(f'Phase Space for delta={delta}')
  plt.xlabel('U_total')
  plt.ylabel('V_total')
  plt.show()

# plot(200,U,V)
# phasespace(U,V,200,dt,dx)

#%%

progbar2=tqdm(total=len(U), desc='anim_loading')


def plot_anim(dt, array1, array2, save_as=None):
    fig, ax = plt.subplots()
    line1, = ax.plot(array1[0, :], label='prey')
    line2, = ax.plot(array2[0, :], linestyle='--', label='predator')
    plt.xlabel('Position')
    plt.ylabel('Population Density')
    plt.grid(True)
    plt.legend()

    def animate(i):
        line1.set_ydata(array1[i, :])
        line2.set_ydata(array2[i, :])
        ax.set_title(f'Population distribution at time={i * dt}')
        progbar2.update(1)
        return line1, line2
        

    ani = animation.FuncAnimation(fig, animate, frames=len(array1), interval=10, blit=True)
        
    if save_as:
      ani.save(save_as, writer='pillow', fps=100)
    
    plt.show()

plot_anim(dt=0.01, array1=U, array2=V,save_as='0.5996_3.gif')
progbar2.close()