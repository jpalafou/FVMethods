import numpy as np
import matplotlib.pyplot as plt
from csv import writer
from util.advection1d import AdvectionSolver

# configurations
order_list = [1, 2, 3, 4, 5, 6]
n_list = [16, 32, 64, 128, 256, 512]
u0_preset = "sinus"
courant = 0.5
T = 2

# file locations
plot_path = "figures/"
data_path = "data/"
project_name = "error_convergence_1d_advection"


def plot_slope_triangle(x, y):
    """
    add right triangles defined by two points to show a slope
    """
    plt.plot(
        [x[0], x[1], x[1]],
        [y[0], y[0], y[1]],
        "-k",
        alpha=0.3,
    )
    slope = (np.log(y[1]) - np.log(y[0])) / (np.log(x[1]) - np.log(x[0]))
    plt.text(
        x[1],
        np.sqrt(y[1] * y[0]),
        f"{round(slope, 2)}",
        horizontalalignment="left",
        verticalalignment="center",
    )


# initialize csv file
with open(data_path + project_name + ".csv", "w+") as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(["label", "h", "error"])
    f_object.close()


# initialize plot
plt.figure(figsize=(12, 8))
triangle_index = 2  # initialize index for forming triangles
for order in sorted(order_list):
    print(f"Solving order {order}")
    h_list = []
    error_list = []
    for n in sorted(n_list):
        print(f"\tn = {n}")
        # initialize solution
        solution = AdvectionSolver(
            n=n,
            order=order,
            u0_preset=u0_preset,
            courant=courant,
            adujst_time_step=True,
            T=T,
        )
        solution.rkorder()  # time integration
        # append error to list of error
        error = solution.find_error("l1")
        h_list.append(solution.h)
        error_list.append(error)
        # label generation
        if solution.order == 1:
            time_message = "euler"
        elif solution.order > 1 and solution.order < 4:
            time_message = f"rk{solution.order}"
        elif solution.order >= 4:
            time_message = "rk4"
        if solution.adujst_time_step and solution.order > 4:
            time_message += (
                " + "
                + r"$\Delta t$"
                + f" * {round(solution.Dt_adjustment, 5)}"
            )
        limiter_message = (
            solution.apriori_limiting if solution.apriori_limiting else "no"
        ) + " limiter"
        limiter_message += (
            " with smooth extrema detection"
            if solution.smooth_extrema_detection
            else ""
        )
        label = (
            f"{limiter_message} order {solution.order}" f" + {time_message}"
        )
        # data logging
        with open(data_path + project_name + ".csv", "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([label, solution.h, error])
            f_object.close()
    # plot the error on loglog
    plt.loglog(h_list, error_list, "-*", label=label)
    # if the order of the solution is even, give it a slope triangle
    if solution.order % 2 == 0:
        plot_slope_triangle(
            [h_list[triangle_index], h_list[triangle_index - 1]],
            [error_list[triangle_index], error_list[triangle_index - 1]],
        )
        if triangle_index == len(h_list) - 3:
            triangle_index = 2
        else:
            triangle_index += 1

# finish plot
plt.xlabel(r"$h$")
plt.ylabel("L1 error")
plt.title(f"C = {courant}, {T} orbits of advection")
plt.legend()
plt.savefig(plot_path + project_name + ".png", dpi=300)
plt.show()
