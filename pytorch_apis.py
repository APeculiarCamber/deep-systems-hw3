import torch as th
import gp_apis

class customlinear_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, device0):
        proper_mult = input.shape[1] == weight.shape[0]
        proper_add = bias.shape[0] == weight.shape[1]
        proper_device = input.is_cuda and weight.is_cuda and bias.is_cuda and weight.device == device0
        if not proper_add or not proper_mult or not proper_device:
            print("Custom Linear Input Error: Bad Formatting or Bad Device", proper_add, proper_mult, proper_device)
            print(f"({input.shape[0]} x {input.shape[1]}) * ({weight.shape[0]} x {weight.shape[1]}) + ({bias.shape[0]}) in {device0}")
            raise Exception("Custom Linear Input Error: Bad Formatting or Bad Device")
        dim1_0, dim1_1 = input.shape[0], weight.shape[1]
        res = gp_apis.gp_customlinear(input, weight, bias, dim1_0, dim1_1, device0)
        ctx.save_for_backward(input, weight)
        ctx.device = device0
        return res

    @staticmethod
    def backward(ctx, dZ):
        input, weight = ctx.saved_tensors
        dim1_0, dim1_1 = dZ.shape[0], weight.shape[0]
        dim2_0, dim2_1 = input.shape[1], dZ.shape[1]        
        dim3_0 = dZ.shape[1]
        dX, dW, dB = gp_apis.gp_customlinear_back(dZ, input, weight, dim1_0, dim1_1, dim2_0, dim2_1, dim3_0, ctx.device)
        return dX, dW, dB, None

def customlinear(input, weight, bias, device0):
    return customlinear_impl.apply(input, weight, bias, device0)




'''
class customlinear_back_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, input3, dim1_0, dim1_1, dim2_0, dim2_1, dim3_0, device0):
        ctx.backward_cache = None #must be implemented
        return res1, res2, res3

    @staticmethod
    def backward(ctx, dZ1, dZ2, dZ3):
        pass #must be implemented

def customlinear_back(input1, input2, input3, dim1_0, dim1_1, dim2_0, dim2_1, dim3_0, device0):
    return customlinear_back_impl.apply(input1, input2, input3, dim1_0, dim1_1, dim2_0, dim2_1, dim3_0, device0)

'''