import torch
import my_extension


class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # cache input for backward
        return my_extension.square_cuda(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return 2 * input * grad_output  # chain rule: d(x^2)/dx = 2x

x = torch.randn(1000, device='cuda')  # example input
y = my_extension.square_cuda(x)
print(y.shape, y.device, y.dtype)  # should match input, with each element squared