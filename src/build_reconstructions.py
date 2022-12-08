import csv
import os.path
from util.fvscheme import Kernel, Interpolation

save_path = "src/util/reconstruction_schemes/"


# devise a scheme for reconstructed values at cell interfaces
orders = range(1,10)


for order in orders:
    if order % 2 != 0: # odd order
        kern = Kernel(order // 2, order // 2) 
        right_interface_scheme = Interpolation.construct(kern, 'r')
        left_interface_scheme = Interpolation.construct(kern, 'l')
    else: # even order
        l = order // 2 # long length
        s = order // 2 - 1 # short length
        right_interface_scheme = (Interpolation.construct(Kernel(l, s), 'r') + Interpolation.construct(Kernel(s, l), 'r')) / 2
        left_interface_scheme = (Interpolation.construct(Kernel(l, s), 'l') + Interpolation.construct(Kernel(s, l), 'l')) / 2
    save_path_right = save_path + f"order{order}_right.csv"
    save_path_left = save_path + f"order{order}_left.csv"

    # right scheme
    with open(save_path_right, 'w+') as file_right:
        writer = csv.writer(file_right)
        for key, val in right_interface_scheme.coeffs.items():
            writer.writerow([key, val])
    print(f"Wrote a right interface reconstruction scheme of order {order} to {save_path_right}")

    # right scheme
    with open(save_path_left, 'w+') as file_left:
        writer = csv.writer(file_left)
        for key, val in left_interface_scheme.coeffs.items():
            writer.writerow([key, val])
    print(f"Wrote a right interface reconstruction scheme of order {order} to {save_path_left}")