import numpy as np
import matplotlib.pyplot as plt
from csv import writer
from util.integrate import rk4_Dt_adjust
from util.initial_condition import initial_condition2d
import util.advection2d as advection2d


# inputs
ic_type = "sinus"  # initial condition type
a = [1, 1]  # tranpsort speed
x_bounds = [0, 1]  # spatial domain
T = 2  # solving time

# admin
plot_path = "figures/"
data_path = "data/"
project_name = "error_convergence_2d_advection"

# configurations
global_configs = {
    "mesh sizes": [16, 32, 64],
    "error norm": "l1",
    "solution scheme": "no limiter",
    "courant factor": 0.5,
}
solution_configs = [
    {
        "spatial order": 1,
        "time order": 1,
        "courant factor": 0.5,
        "solution scheme": "mpp",
    },
    {
        "spatial order": 2,
        "time order": 2,
        "courant factor": 0.5,
        "solution scheme": "mpp",
    },
    {
        "spatial order": 3,
        "time order": 3,
        "courant factor": 0.5,
        "solution scheme": "mpp",
    },
    {
        "spatial order": 4,
        "time order": 4,
        "courant factor": 0.5,
        "solution scheme": "mpp",
    },
    {
        "spatial order": 5,
        "time order": 5,
        "courant factor": 0.5,
        "solution scheme": "mpp",
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

# start plot
plt.figure(figsize=(12, 8))
# initialize index for forming triangles
triangle_index = 1
h_list = [
    (x_bounds[1] - x_bounds[0]) / n for n in global_configs["mesh sizes"]
]
for config in solution_configs:
    errors = []
    for n in global_configs["mesh sizes"]:
        # global config
        for key in config.keys():
            if key in global_configs.keys():
                config[key] = global_configs[key]
        # print update
        for key, item in config.items():
            print(f"{key}: {item}")
        print(f"mesh size: {n}")
        print()
        # array of x-values
        x_interface = np.linspace(x_bounds[0], x_bounds[1], num=n + 1)
        x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers
        h = (x_bounds[1] - x_bounds[0]) / n
        # time vector
        time_step_adjustment_label = ""
        rkorder = config["time order"]
        Dt = config["courant factor"] * h / max(np.abs(a))
        if config["time order"] > 4:
            rkorder = 4
            time_step_adjustment = rk4_Dt_adjust(
                h, x_bounds[1] - x_bounds[0], config["spatial order"]
            )
            Dt = Dt * time_step_adjustment
            time_step_adjustment_label = (
                r" + $\Delta t$ * " + f"{round(time_step_adjustment, 5)}"
            )
        n_time = int(np.ceil(T / Dt))
        t = np.linspace(0, T, num=n_time)
        # initial condition
        u0 = initial_condition2d(x, ic_type)
        # set up solution
        if config["solution scheme"] == "no limiter":
            solution = advection2d.AdvectionSolver(
                u0=u0, t=t, h=h, a=a[0], b=a[1], order=config["spatial order"]
            )
        else:
            raise BaseException(
                f"invalid solution scheme {config['solution scheme']}"
            )
        # time integration
        solution.rkn(rkorder)
        # find error
        if global_configs["error norm"] == "l1":
            error = np.sum(np.abs((solution.u[-1] - u0))) / u0.size
        else:
            raise BaseException(
                f"invalid error norm {global_configs['error norm']}"
            )
        # append error to list of errors
        errors.append(error)
        # label generation
        if rkorder == 1:
            time_message = "euler"
        else:
            time_message = f"rk{rkorder}{time_step_adjustment_label}"
        label = (
            f"{config['solution scheme']} order {config['spatial order']}"
            f" + {time_message}"
        )
        # data logging
        with open(data_path + project_name + ".csv", "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([label, h, error])
            f_object.close()
    # plotting
    # plot the error on loglog
    plt.loglog(h_list, errors, "-*", label=label)
    # if the order of the solution is even, give it a slope triangle
    if config["spatial order"] % 2 == 0:
        plot_slope_triangle(
            [h_list[triangle_index], h_list[triangle_index - 1]],
            [errors[triangle_index], errors[triangle_index - 1]],
        )
        if triangle_index == len(h_list) - 1:
            triangle_index = 1
        else:
            triangle_index += 1

# finish plot
plt.xlabel(r"$h$")
plt.ylabel(f"{global_configs['error norm']} " + r"$\epsilon$")
C = global_configs["courant factor"]
plt.title(f"C = {C}, {T} orbits of advection")
plt.legend()
plt.savefig(plot_path + project_name + ".png", dpi=300)
