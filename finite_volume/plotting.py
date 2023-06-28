import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcol

colors = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
}

color_list = list(colors.values())


def lineplot(
    solution_dictionary_in,
    x: float = None,
    y: float = None,
    show: bool = True,
    savepath: str = None,
):
    if isinstance(solution_dictionary_in, dict):
        solution_dictionary = solution_dictionary_in.copy()
    else:
        solution_dictionary = {"data1": solution_dictionary_in}

    plt.figure()
    first_solution = list(solution_dictionary.values())[0]
    if first_solution.ndim == 1:
        # plot t = 0
        plt.plot(first_solution.x, first_solution.u[0], label="t = 0")
        # plot all curves
        for label, solution in solution_dictionary.items():
            plt.plot(solution.x, solution.u[-1], "o--", mfc="none", label=label)
        plt.xlabel("x")
    if first_solution.ndim == 2:
        if x is not None:
            plot_direction = "x"
        if y is not None:
            plot_direction = "y"
        # plot t = 0
        if plot_direction == "x":
            xdata = first_solution.y
            idx = np.where(first_solution.x == x)[0]
            if idx.size != 1:
                raise BaseException("x slice is not of size 1")
            ydata = first_solution.u[0][:, idx]
            plt.xlabel("y")
        if plot_direction == "y":
            xdata = first_solution.x
            idx = np.where(first_solution.y == y)[0]
            if idx.size != 1:
                raise BaseException("y slice is not of size 1")
            ydata = first_solution.u[0][idx, :].flatten()
            plt.xlabel("x")
        plt.plot(xdata, ydata, label="t = 0")
        # plot all curves
        for label, solution in solution_dictionary.items():
            if plot_direction == "x":
                xdata = solution.y
                idx = np.where(first_solution.x == x)[0]
                if idx.size != 1:
                    raise BaseException("x slice is not of size 1")
                ydata = solution.u[-1][:, idx]
            if plot_direction == "y":
                xdata = solution.x
                idx = np.where(first_solution.y == y)[0]
                if idx.size != 1:
                    raise BaseException("y slice is not of size 1")
                ydata = solution.u[-1][idx, :].flatten()
            plt.plot(xdata, ydata, "o--", mfc="none", label=label)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    if show:
        plt.show()


def heatmap(solution, fig=None, decorate=False, show=False):
    # global max and min
    bounds = [solution.x[0], solution.x[-1], solution.y[0], solution.y[-1]]

    # select plot indices for the first 5 frames
    idx1 = 0
    idx2 = int(0.25 * solution.loglen)
    idx3 = int(0.5 * solution.loglen)
    idx4 = int(0.75 * solution.loglen)
    idx5 = -1

    # gather data
    hmin, hmax = np.min(solution.u), np.max(solution.u)
    u = solution.u
    data1 = np.flipud(u[idx1])
    data2 = np.flipud(u[idx2])
    data3 = np.flipud(u[idx3])
    data4 = np.flipud(u[idx4])
    data5 = np.flipud(u[idx5])
    data6 = np.flipud(solution.u[idx5] - solution.u[idx1])

    # if data == "limiting" and self.apriori_limiting:
    #     hmin, hmax = 0, 1
    #     theta_history = np.where(
    #         self.visualize_theta_history, self.theta_history, 1
    #     )
    #     data1 = np.flipud(1 - theta_history[idx1])
    #     data2 = np.flipud(1 - theta_history[idx2])
    #     data3 = np.flipud(1 - theta_history[idx3])
    #     data4 = np.flipud(1 - theta_history[idx4])
    #     data5 = np.flipud(1 - theta_history[idx5])
    #     data6 = None
    # if data == "limiting" and self.aposteriori_limiting:
    #     if self.apriori_limiting:
    #         print("Warning: Theta data overwritten by troubled cell data.")
    #     hmin, hmax = 0, 1
    #     troubled_cell_history = np.where(
    #         self.visualize_troubled_cell_history, self.troubled_cell_history, 0
    #     )
    #     data1 = np.flipud(troubled_cell_history[idx1])
    #     data2 = np.flipud(troubled_cell_history[idx2])
    #     data3 = np.flipud(troubled_cell_history[idx3])
    #     data4 = np.flipud(troubled_cell_history[idx4])
    #     data5 = np.flipud(troubled_cell_history[idx5])
    #     data6 = None

    # 2x3 grid of subplots
    fig = plt.figure(figsize=(10, 6.5))
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=4,
        width_ratios=[1, 1, 1, 0.1],
        height_ratios=[1, 1],
        figure=fig,
    )
    axs = gs.subplots()

    # custom coloring
    wtc = mcol.LinearSegmentedColormap.from_list("wtc", ["white", "blue"])

    im1 = axs[0, 0].imshow(data1, cmap=wtc, vmin=hmin, vmax=hmax, extent=bounds)
    axs[0, 1].imshow(data2, cmap=wtc, vmin=hmin, vmax=hmax, extent=bounds)
    axs[0, 2].imshow(data3, cmap=wtc, vmin=hmin, vmax=hmax, extent=bounds)
    axs[1, 0].imshow(data4, cmap=wtc, vmin=hmin, vmax=hmax, extent=bounds)
    axs[1, 1].imshow(data5, cmap=wtc, vmin=hmin, vmax=hmax, extent=bounds)

    # first color bar
    cbar1 = fig.add_subplot(gs[0, 3])
    plt.colorbar(im1, cax=cbar1)
    cbar1.yaxis.set_ticks_position("right")

    # Add titles to the plots
    axs[0, 0].set_title(f"t = {solution.t[idx1]:.3f}")
    axs[0, 1].set_title(f"t = {solution.t[idx2]:.3f}")
    axs[0, 2].set_title(f"t = {solution.t[idx3]:.3f}")
    axs[1, 0].set_title(f"t = {solution.t[idx4]:.3f}")
    axs[1, 1].set_title(f"t = {solution.t[idx5]:.3f}")

    if data6 is not None:
        maxerror = np.max(np.abs(data6))
        im6 = axs[1, 2].imshow(
            data6, cmap="seismic", vmin=-maxerror, vmax=maxerror, extent=bounds
        )
        axs[1, 2].set_title(
            f"u(t = {solution.t[idx5]:.3f}) - u(t = {solution.t[idx1]:.3f})"
        )
        # second color bar
        cbar2 = fig.add_subplot(gs[1, 3])
        plt.colorbar(im6, cax=cbar2)
        cbar2.yaxis.set_ticks_position("right")

    # Show the plot
    plt.show()


def contour(solution):
    # Heights at which the contour curves will be drawn
    heights = np.linspace(0.1, 0.9, 9)

    # Create a rainbow colormap
    cmap = plt.cm.get_cmap("rainbow")

    # Create the contour plot
    plt.contour(solution.x, solution.y, solution.u[-1], levels=heights, cmap=cmap)

    # Show the plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.show()


def cube(solution, k: int = -1):
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x, y = np.meshgrid(solution.x, solution.y)
    x, y = x.flatten(), y.flatten()
    dx, dy, = (
        solution.hx,
        solution.hy,
    )
    dz = solution.u[k].flatten()
    ax.bar3d(x, y, 0, dx, dy, dz, color="white", edgecolor="black", linewidth=0.1)

    # Set labels and title
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")

    # Show the plot
    ax.set_box_aspect([1, 1, 0.5])  # Adjust the scaling factors as desired
    plt.show()


def minmax(
    solution_dictionary_in,
    show: bool = True,
    zeroline: bool = True,
    savepath: str = None,
):
    if isinstance(solution_dictionary_in, dict):
        solution_dictionary = solution_dictionary_in.copy()
    else:
        solution_dictionary = {"data1": solution_dictionary_in}

    fig, axs = plt.subplots(2, 1, sharex="col", figsize=(10, 6))

    first_solution = list(solution_dictionary.values())[0]

    if zeroline:
        T = first_solution.t[-1]
        axs[0].plot([0, T], [0, 0], "--", color="grey")
        axs[1].plot([0, T], [0, 0], "--", color="grey")

    idx = 0
    for label, solution in solution_dictionary.items():
        if solution.ndim == 1:
            minaxis = 1
        elif solution.ndim == 2:
            minaxis = (1, 2)
        if label[:2] == "--":
            linestyle = "--"
            current_label = label[2:]
        else:
            linestyle = "-"
            current_label = label
        axs[0].plot(
            solution.t,
            np.amax(solution.u, axis=minaxis) - 1,
            color=color_list[idx],
            linestyle=linestyle,
            label=current_label,
        )
        axs[1].plot(
            solution.t,
            np.amin(solution.u, axis=minaxis),
            color=color_list[idx],
            linestyle=linestyle,
            label=current_label,
        )
        idx += 1
    axs[0].set_ylabel("max(u(t)) - 1")
    axs[1].set_ylabel("min(u(t))")
    axs[1].set_xlabel("t")
    axs[1].legend()

    if savepath:
        plt.savefig(savepath, dpi=300)

    if show:
        plt.show()
