from util.fvscheme import PolynomialReconstruction

orders = [5, 6, 7, 8, 9, 10]

for order in orders:
    PolynomialReconstruction.construct_from_order(
        order=order, reconstruct_here="right"
    )
    PolynomialReconstruction.construct_from_order(
        order=order, reconstruct_here="left"
    )
