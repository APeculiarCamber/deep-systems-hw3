import torch as th
import torch.utils.dlpack
import graphpy as gpk
def gp_customlinear(input1, input2, input3, dim1_0, dim1_1, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    input3_dl = th.utils.dlpack.to_dlpack(input3)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.customlinear(input1_dl, input2_dl, input3_dl, res_dl1)
    return res1

def gp_customlinear_back(input1, input2, input3, dim1_0, dim1_1, dim2_0, dim2_1, dim3_0, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    input3_dl = th.utils.dlpack.to_dlpack(input3)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    res2 = th.zeros(dim2_0, dim2_1, device = device0)
    res_dl2 = th.utils.dlpack.to_dlpack(res2)
    res3 = th.zeros(dim3_0, device = device0)
    res_dl3 = th.utils.dlpack.to_dlpack(res3)
    gpk.customlinear_back(input1_dl, input2_dl, input3_dl, res_dl1, res_dl2, res_dl3)
    return res1, res2, res3
