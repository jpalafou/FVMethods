from configs import limiting_schemes_2d, problem_configs, timing_solver_config
from finite_volume.advection import AdvectionSolver

limiting_schemes = limiting_schemes_2d

n = 512
p = 7
n_steps = 10
lckey = "aPosterioriB"
cupy = True

solver = AdvectionSolver(
    n=(n,),
    order=p + 1,
    cupy=cupy,
    snapshot_dt=1.0,
    num_snapshots=1,
    **limiting_schemes[lckey],
    **problem_configs["square2d"],
    **timing_solver_config,
    load=False,
)
solver.snapshot_dt = 0
for _ in range(n_steps):
    solver.snapshot_dt += solver.dt
solver.rkorder(rk6=True if p > 3 else False)

print(f"{solver.solution_time=}")
