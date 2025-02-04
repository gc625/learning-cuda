

All threads in a block are physically in the same streaming multiprocessor core, as a result, there is a limit to how many threads can be in a block.

- currently this limit is 1024 theads per block.

Q: does this mean the `Dx * Dy * Dz` <= 1024?



But a kernal can be executed by many thread blocks with the **same** shape. Blocks are organized into 1-3 dim **grids**.

So suppose we have `Bx * By * Bz` blocks. since they are for the same kernel, their threads must also be organized in the same way, say `Dx * Dy * Dz`. Then the *total* threads actually executing your kernel is `Bx * By * Bz * Dx * Dy * Dz`.

Q: how many blocks can we make in a grid?


Q: are all thread blocks allocated for a kernel guaranteed to be in the same multiprocessor core?

A: figure 3 from the article, no it does not. we see that for GPUs with varying number of SMs, blocks are scheduled differently.  
![image](fig_3_automatic-scalability.png)



