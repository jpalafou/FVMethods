from util.fvscheme import PolynomialReconstruction

orders = range(1, 10)

for order in orders:
    PolynomialReconstruction.construct_from_order(
        order=order, reconstruct_here="right"
    )
    PolynomialReconstruction.construct_from_order(
        order=order, reconstruct_here="left"
    )
