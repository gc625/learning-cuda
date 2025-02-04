import numpy as np
from utils import *
np.random.seed(42)

def flatten_idx(column_index, row_index, num_columns, num_rows):
    """
    row_index (x): which row?  
    column_index (y): which column within the row? 
    
    num_columns (Dx): how many columns?
    num_rows (Dy): how many rows?

    """
            # x             y       * Dx 
    return column_index + (row_index * num_columns)

def unflatten_idx(flat_idx, num_columns, num_rows):
    
    
    
    x = flat_idx % num_columns
    y = flat_idx // Dx
    return x, y

def main_flattening(Dx,Dy):
    # Create a "thread block" of shape [Dx, Dy]
    thread_block = np.random.rand(Dy, Dx)
    thread_block_flat = thread_block.flatten()

    pprint_np(thread_block, f"{Dy} rows with {Dx} columns", print_color=True)
    pprint_flat(thread_block_flat, thread_block, "flat thread block")


    x,y = 3,2
    # x=3 and y=2 means I want the (3+1)th elem in the row, in the (2+1)rd row

    flat_idx = flatten_idx(x,y,Dx,Dy)
    
    print(f"threadblock[{y},{x}]: {round(thread_block[y,x],3)}")
    print(f"x: {x} | y*Dx = {y}*{Dx}: {y*Dx}")
    print(f'threadblock_flat[{flat_idx}]: {round(thread_block_flat[flat_idx],3)}')



def main_unflattening(Dx,Dy):

    thread_block = np.random.rand(Dy, Dx)
    thread_block_flat = thread_block.flatten()

    pprint_np(thread_block, f"{Dy} rows with {Dx} columns", print_color=True)
    pprint_flat(thread_block_flat, thread_block, "flat thread block")

    flat_idx = 13

    # because we know the size of the threadBlock, we can infer the x and y

    x, y = unflatten_idx(flat_idx,num_columns=Dx,num_rows=Dy)

    print(f'threadblock_flat[{flat_idx}]: {round(thread_block_flat[flat_idx],3)}')
    print(f"threadblock[{y},{x}]: {round(thread_block[y,x],3)}")
    print(f"x: {x} | y*Dx = {y}*{Dx}: {y*Dx}")






if __name__ == '__main__':
    Dx = 5 # num of columns
    Dy = 3 # num of rows  

    # main_flattening(Dx,Dy)
    main_unflattening(Dx,Dy)