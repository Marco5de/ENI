#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt

def lin_model(tau,weight): 
    #create time vector with start tb=0 te=30 and dt=0.1
    tb=0
    te=30.1
    dt=0.1
    
    t=np.arange(0, 30.1, 0.1)
    #create x1 rect-function
    x1=[]
    for t_cur in t:
        if t_cur<5 or t_cur>15:
            x1.append(0)
        else:
            x1.append(1)
            
    #computing u1-vector
    u1=[]
    u1.append(0)
    count=0
    for t_cur in t:
        u1.append(u1[count]*(1-dt/tau)+dt/tau*x1[count])
        count+=1
    #computing u2-vector    
    veclen=count
    u2=[]
    u2.append(0)
    count=0
    for t_cur in t:
        u2.append(u2[count]*(1-dt/tau)+dt/tau*weight*u1[count])
        count+=1
        
    #computing u1/2'-vector
    du1=[]
    du2=[]
    for i in range(0,veclen):
        du1.append(1/dt*(u1[i+1]-u1[i]))
        du2.append(1/dt*(u2[i+1]-u2[i]))
        
    
    return [t,du1,du2,u1[1:],u2[1:]]
    
#print(du2)


# In[8]:


import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets

def plot_model(tau,weight):
    t,du1,du2,u1,u2=lin_model(tau,weight)
    
    plt.subplot(2,2,1)
    plt.title("Potenzial am Neuron 1")
    plt.plot(t,u1)
    
    plt.subplot(2,2,2)
    plt.title("Potenzial am Neuron 2")
    plt.plot(t,u2)
    
    plt.subplot(2,2,3)
    plt.title("Änderung des Potenzials am Neuron 1")
    plt.plot(t,du1)
    
    plt.subplot(2,2,4)
    plt.title("Änderung des Potenzials am Neuron 2")
    plt.plot(t,du2)
    
    plt.subplots_adjust(left=None, bottom=3, right=5, top=5, wspace=None, hspace=None)
    
    plt.show()

    
    
    
    
#plot_model(1,0.75) 
interact(plot_model, tau=(0.1,10,0.1),weight=(0,1,0.05));


    


# In[ ]:




