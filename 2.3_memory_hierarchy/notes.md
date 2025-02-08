- Each thread has private local memory
- All threads in a thread block with same lifetime has shared memory
- Thread blocks can perform read/write/atomic ops on eachother's shared memory (compute capability 9.0)
- All threads can access global memory.

There are two additional read only memory spaces:
1. Constant memory space
2. Texture memory space

global, constant and texture memory are optimized for different usages. They also persist across kernel launches by the same application

![image](memory-hierarchy.png)