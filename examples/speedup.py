import matplotlib.pyplot as plt
from finite_volume.advection import AdvectionSolver

u0 = "square"
T = 2
v = (2, 1)
load = False
num_trials = 3
n_list = [16, 32, 64]
order_list = [1, 3, 4, 5]

mpp_cfl = {1: 0.5, 2: 0.5, 3: 0.166, 4: 0.166, 5: 0.0833}

for order in order_list:
    print(f"order {order}")
    speedup_list = []
    for n in n_list:
        print(f"\tn = {n}")
        normal_solving_time = 0
        fast_solving_time = 0
        for i in range(num_trials):
            print(f"\t\ttrial {i + 1}")
            normal_solution = AdvectionSolver(
                u0=u0,
                T=T,
                v=v,
                n=(n,),
                order=order,
                modify_time_step=False,
                courant=mpp_cfl[order],
                flux_strategy="gauss-legendre",
                apriori_limiting=True,
                aposteriori_limiting=False,
                smooth_extrema_detection=False,
                load=load,
            )
            normal_solution.rkorder()
            normal_solving_time += normal_solution.solution_time

            fast_solution = AdvectionSolver(
                u0=u0,
                T=T,
                v=v,
                n=(n,),
                order=order,
                modify_time_step=True,
                courant=0.8,
                flux_strategy="gauss-legendre",
                apriori_limiting=True,
                aposteriori_limiting=False,
                smooth_extrema_detection=False,
                load=load,
            )
            fast_solution.rkorder()
            fast_solving_time += fast_solution.solution_time
        speedup_list.append(normal_solving_time / fast_solving_time)
    plt.plot(n_list, speedup_list, "-*", label=f"order {order}")
plt.legend()
plt.show()
