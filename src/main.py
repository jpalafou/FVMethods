import numpy as np
import math
import matplotlib.pyplot as plt
from util.fvscheme import Kernel, Interpolation

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

# devise a scheme
order = 2

if order % 2 != 0: # odd order
    kern_left = Kernel(order // 2, order // 2, -1)
    kern_center = Kernel(order // 2, order // 2)
    kern_right = Kernel(order // 2, order // 2, 1)
    right_winded_flux = Interpolation.construct(kern_center, 'r') - Interpolation.construct(kern_left, 'r')
    left_winded_flux = Interpolation.construct(kern_right, 'l') - Interpolation.construct(kern_center, 'l')
else: # even order
    l = order // 2 # long length
    s = order // 2 - 1 # short length
    right_winded_flux = (Interpolation.construct(Kernel(l, s), 'r') + Interpolation.construct(Kernel(s, l), 'r')) / 2 - (Interpolation.construct(Kernel(l, s, -1), 'r') + Interpolation.construct(Kernel(s, l, -1), 'r')) / 2
    left_winded_flux = (Interpolation.construct(Kernel(l, s, 1), 'l') + Interpolation.construct(Kernel(s, l, 1), 'l')) / 2 - (Interpolation.construct(Kernel(l, s), 'l') + Interpolation.construct(Kernel(s, l), 'l')) / 2


print(right_winded_flux.nparray())

# get a scalar value for right and left flux BEFORE i take the difference
# this step is post-reiman
