import numpy as np
import matplotlib.pyplot as plt
import math
from fvscheme import FVscheme

a = 3 # choose a positive value of the 'wind' direction

# domain
x_bounds = [-0.5, 2]
h = 0.005
T = 0.5
Dt = 0.9*h/a


# array of x-values
x = np.arange(x_bounds[0], x_bounds[1] + h, h)

# array of t-values
t = np.arange(0, T + Dt, Dt)

# initial values of u
u0 = [np.cos(5*i) if i > -math.pi/10 and i < math.pi/10 else 0 for i in x]

# create field for u
field = FVscheme(x,t,u0)

# solve
field.solve_advection(a,2)
field.interpolation.show()

# plotting
plt.plot(x,field.u[:,0])
plt.plot(x,field.u[:,1])
plt.plot(x,field.u[:,10])
plt.plot(x,field.u[:,100])
plt.show()
