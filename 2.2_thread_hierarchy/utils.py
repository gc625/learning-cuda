import colorama
import numpy as np


colorama.init(autoreset=True)

COLORS = [
    colorama.Fore.RED,
    colorama.Fore.GREEN,
    colorama.Fore.YELLOW,
    colorama.Fore.BLUE,
    colorama.Fore.MAGENTA,
    colorama.Fore.CYAN,
    colorama.Fore.WHITE
]

def pprint_np(np_array, array_name='', print_color=False):
    start_str = "="*5+ array_name+ "="*5 
    print(start_str)
    
    if print_color and np_array.ndim == 2:
        print("[",end='')
        for row_idx, row in enumerate(np_array):
            color = COLORS[row_idx % len(COLORS)]  # Cycle through colors
            row_str = " ".join(f"{color}{x:.3f}" for x in row)
            print(f"[{row_str}]",end='') if row_idx + 1 == len(np_array) else print(f"[{row_str}]") 
        print("]")
    else:
        str_p = np.array2string(np_array, precision=3)
        print(str_p)

    print("="*len(start_str))

def pprint_flat(flat_array, original_array,array_name):
    Dx = original_array.shape[1]
    start_str = "="*5+ array_name+ "="*5 
    print(start_str)
    for idy, y in enumerate(flat_array):
        color = COLORS[(idy //Dx) % len(COLORS)]
        print(f"{color}{y:.3f}", end=" ")
    print()
    print("="*len(start_str))