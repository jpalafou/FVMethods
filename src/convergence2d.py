import numpy as np
import matplotlib.pyplot as plt
from csv import writer
from util.advection2d import AdvectionSolver

# admin
plot_path = "figures/"
data_path = "data/"
project_name = "error_convergence_2d_advection"

# configurations
global_config = {
    "mesh sizes": [16, 32, 64],
    "error norm": "l1",
    "solution scheme": "no limiter",
    "courant factor": 0.5,
    "advection velocity": (1, 1),
    "initial condition": "sinus",
    "solution time": 2,
}
solution_configs = [
    {
        "order": 1,
    },
    {
        "order": 2,
    },
    {
        "order": 3,
    },
    {
        "order": 4,
    },
    {
        "order": 5,
    },
]


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
triangle_index = 0  # initialize index for forming triangles
for config in solution_configs:
    h_list = []
    error_list = []
    for n in global_config["mesh sizes"]:
        # global config
        for key in global_config.keys():
            config[key] = global_config[key]
        # print update
        print("- - - - -")
        for key, item in config.items():
            print(f"{key}: {item}")
        print(f"mesh size: {n}")
        print()
        # define solution
        if config["solution scheme"] == "no limiter":
            solution = AdvectionSolver(
                u0_preset=config["initial condition"],
                n=n,
                T=config["solution time"],
                a=config["advection velocity"],
                courant=config["courant factor"],
                order=config["order"],
            )
        else:
            raise BaseException(
                f"invalid solution scheme {config['solution scheme']}"
            )
        # time integration
        solution.rkn(config["order"])
        # find error
        if config["error norm"] == "l1":
            error = np.sum(
                np.abs(solution.u[-1] - solution.u[0])
                / (solution.nx * solution.ny)
            )
        else:
            raise BaseException(f"invalid error norm {config['error norm']}")
        # append error to list of errors
        h_list.append(solution.hx)
        error_list.append(error)
        # label generation
        if config["order"] == 1:
            time_message = "euler"
        elif config["order"] > 1 and config["order"] <= 4:
            time_message = f"rk{config['order']}"
        else:
            time_message = (
                "rk4 + "
                + r"$\Delta t$"
                + f" * {round(solution.time_step_adjustment, 5)}"
            )
        label = (
            f"{config['solution scheme']} order {config['order']}"
            f" + {time_message}"
        )
        # data logging
        with open(data_path + project_name + ".csv", "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([label, solution.hx, error])
            f_object.close()
    # plot the error on loglog
    plt.loglog(h_list, error_list, "-*", label=label)
    # if the order of the solution is even, give it a slope triangle
    if config["order"] % 2 == 0:
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
plt.ylabel(f"{global_config['error norm']} " + r"$\epsilon$")
C = global_config["courant factor"]
T = global_config["solution time"]
plt.title(f"C = {C}, {T} orbits of advection")
plt.legend()
plt.savefig(plot_path + project_name + ".png", dpi=300)
plt.show()
