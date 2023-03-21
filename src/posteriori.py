import matplotlib.pyplot as plt
from util.advection1d import AdvectionSolver

solution = AdvectionSolver(n=64, order=4, u0_preset="square", aposteriori=True)
solution.rk4()

plt.plot(solution.x, solution.u[0])
plt.plot(solution.x, solution.u[-1])
plt.show()
