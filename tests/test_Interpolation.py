import pytest
import numpy as np
from src import fvscheme
from src.fvscheme import Kernel
from src.fvscheme import Interpolation

# Teyssier has /Delta x seperate from his expression for du. mine is built in to
# the solution for u as a weighted average of cell values. is this a problem?

def test_2nd_order_central_difference():
    du_solution = {-1:-1/2, 1:1/2}
    my_du = (Interpolation(Kernel(0,1),'r') - Interpolation(Kernel(1,0),'l')).u_weights
    assert my_du == du_solution


def test_2nd_order_Fromm():
    right_solution = {-1:-1/4, 0:1, 1:1/4}
    right_average = (Interpolation(Kernel(0,1),'r') + Interpolation(Kernel(1,0),'r'))/2
    assert right_average.u_weights == right_solution

    left_solution = {-1:1/4, 0:1, 1:-1/4}
    left_average = (Interpolation(Kernel(0,1),'l') + Interpolation(Kernel(1,0),'l'))/2
    assert left_average.u_weights == left_solution

    du_solution = {-2:1/4, -1:-5/4, 0:3/4, 1:1/4}
    my_du = ((Interpolation(Kernel(0,1),'r') + Interpolation(Kernel(1,0),'r'))/2 - \
    (Interpolation(Kernel(0,1,-1),'r') + Interpolation(Kernel(1,0,-1),'r'))/2).u_weights
    assert my_du == du_solution

def test_3rd_order_difference():
    du_solution = {-2:1/6, -1:-6/6, 0:3/6, 1:2/6}
    my_du = (Interpolation(Kernel(1,1),'r') - Interpolation(Kernel(1,1,-1),'r')).u_weights
    assert my_du == pytest.approx(du_solution)

@pytest.mark.parametrize("unused_parameter", range(5))  # test 5 times
def test_evaluate(unused_parameter):
    # use right solution from test_2nd_order_Fromm
    v = np.random.randint(-10,10,3)
    my_scheme = (Interpolation(Kernel(0,1),'r') + Interpolation(Kernel(1,0),'r'))/2
    assert my_scheme.evaluate(v,1) == (-1/4)*v[0] + v[1] + (1/4*v[2])
