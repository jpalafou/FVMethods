from finite_volume.fvscheme import ConservativeInterpolation

orders = range(1, 8)

ConservativeInterpolation.construct_from_order(
    order=5, reconstruct_here=0.49
)  # this should approximate the right interface reconstruction

for order in orders:
    ConservativeInterpolation.construct_from_order(order=order, reconstruct_here="left")
    ConservativeInterpolation.construct_from_order(
        order=order, reconstruct_here="right"
    )
