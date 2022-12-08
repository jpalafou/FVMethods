import numpy as np
import math
import matplotlib.pyplot as plt
from util.integrate import Integrator
from util.fvscheme import Kernel, Interpolation
# import time
# start_time = time.time()

a = 1 # choose a positive value of the 'wind' direction

# domain
x_bounds = [-0.5, 2]
h = 0.01
T = 2.5
Dt = 0.8*h/a


# array of x-values
x = np.arange(x_bounds[0], x_bounds[1] + h, h)

# array of t-values
t = np.arange(0, T + Dt, Dt)

# initial values of u
# u0 = np.array([np.cos(5*i) if i > -math.pi/10 and i < math.pi/10 else 0 for i in x])
u0 = np.array([1 if i > 0 and i < 0.5 else 0 for i in x])

# devise a scheme for reconstructed values at cell interfaces
order = 1

if order % 2 != 0: # odd order
    kern_left = Kernel(order // 2, order // 2, -1)
    kern_center = Kernel(order // 2, order // 2)
    kern_right = Kernel(order // 2, order // 2, 1)
    if a > 0: # assumes a is constant
        right_interface_scheme = Interpolation.construct(kern_center, 'r')
        left_interface_scheme = Interpolation.construct(kern_left, 'r')
    elif a < 0:
        right_interface_scheme = Interpolation.construct(kern_right, 'l')
        left_interface_scheme = Interpolation.construct(kern_center, 'l')
else: # even order
    l = order // 2 # long length
    s = order // 2 - 1 # short length
    if a > 0:
        right_interface_scheme = (Interpolation.construct(Kernel(l, s), 'r') + Interpolation.construct(Kernel(s, l), 'r')) / 2
        left_interface_scheme = (Interpolation.construct(Kernel(l, s, -1), 'r') + Interpolation.construct(Kernel(s, l, -1), 'r')) / 2
    elif a < 0:
        right_interface_scheme = (Interpolation.construct(Kernel(l, s, 1), 'l') + Interpolation.construct(Kernel(s, l, 1), 'l')) / 2
        left_interface_scheme = (Interpolation.construct(Kernel(l, s), 'l') + Interpolation.construct(Kernel(s, l), 'l')) / 2
right_interface_scheme_np = right_interface_scheme.nparray()
left_interface_scheme_np = left_interface_scheme.nparray()

# right reach of the right interface scheme (and so on)
ri_rr = max(right_interface_scheme.coeffs.keys())
ri_lr = min(right_interface_scheme.coeffs.keys())
li_rr = max(left_interface_scheme.coeffs.keys())
li_lr = min(left_interface_scheme.coeffs.keys())

# prepare for integrator
# du/dt = -a du/dx

print()
print("right interface:")
print(right_interface_scheme)
print("as np array:")
print(right_interface_scheme_np)
print()
print("left interface:")
print(left_interface_scheme)
print("as np array:")
print(left_interface_scheme_np)
print()

class advection_solver(Integrator):
    def xdot(self, x, t_i):
        # # third order
        # # right interface
        # p = right_interface_scheme_np
        # u_right = (np.roll(x, 1)*p[0] + np.roll(x, 0)*p[1] + np.roll(x, -1)*p[2]) / sum(p)
        # # left interface
        # p = left_interface_scheme_np
        # u_left = (np.roll(x, 2)*p[0] + np.roll(x, 1)*p[1] + np.roll(x, 0)*p[2]) / sum(p)
        # return -(a / h) * (u_right - u_left)

        # # fourth order
        # # right interface
        # p = right_interface_scheme_np
        # u_right = np.roll(x, 2)*p[0] + np.roll(x, 1)*p[1] + np.roll(x, 0)*p[2] + np.roll(x, -1)*p[3] + np.roll(x, -2)*p[4]
        # # left interface
        # p = left_interface_scheme_np
        # u_left = np.roll(x, 3)*p[0] + np.roll(x, 2)*p[1] + np.roll(x, 1)*p[2] + np.roll(x, 0)*p[3] + np.roll(x, -1)*p[4]
        # return -(a / h) * (u_right - u_left)

        # right interface reconstruction
        A_r = np.array([np.roll(x, -i) for i in range(ri_lr, ri_rr + 1)]).T
        u_right = A_r @ right_interface_scheme_np / sum(right_interface_scheme_np)
        # left interface reconstruction
        A_l = np.array([np.roll(x, -i) for i in range(li_lr, li_rr + 1)]).T
        u_left = A_l @ left_interface_scheme_np / sum(left_interface_scheme_np)
        return -(a / h) * (u_right - u_left)

advection_solution = advection_solver(u0, t)
advection_solution.rk4()

plt.plot(x, advection_solution.x[:,0], x, advection_solution.x[:,0])
# plt.plot(x, advection_solution.x[:,0], x, advection_solution.x[:,1])
# plt.plot(x, advection_solution.x[:,0], x, advection_solution.x[:,2])
plt.plot(x, advection_solution.x[:,0], x, advection_solution.x[:,int(1 * len(t) / 4)])
plt.plot(x, advection_solution.x[:,0], x, advection_solution.x[:,int(2 * len(t) / 4)])
plt.plot(x, advection_solution.x[:,0], x, advection_solution.x[:,int(3 * len(t) / 4)])
plt.plot(x, advection_solution.x[:,0], x, advection_solution.x[:,-1])
plt.show()

# 1 RECONSTRUCT
# 2 REIMANN SOLVER
# 3 CONSERVATIVE UPDATE

# end_time = time.time()
# print(end_time - start_time)