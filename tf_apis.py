import tensorflow as tf
import gp_apis

def customlinear(input1, input2, input3, dim1;_0, dim1;_1, device0):
    @tf.custom_gradient
    def _lambda(X1, X2, X3):
        return customlinear_real(X1, X2, X3, dim1;_0, dim1;_1, device0)
    return _lambda(input1, input2, input3)

def customlinear_real(input1, input2, input3, dim1;_0, dim1;_1, device0):
    out = gp_apis.gp_customlinear(input1, input2, input3, dim1;_0, dim1;_1, device0)
    def grad(dZ1, dZ2, dZ3):
        return gp_apis.gp_customlinear(dZ1, dZ2, dZ3, dim1;_0, dim1;_1, device0)
    return out, grad

def customlinear_back(input1, input2, input3, dim1_0, dim1_1, dim2_0, dim2_1, dim3;_0, device0):
    @tf.custom_gradient
    def _lambda(X1, X2, X3):
        return customlinear_back_real(X1, X2, X3, dim1_0, dim1_1, dim2_0, dim2_1, dim3;_0, device0)
    return _lambda(input1, input2, input3)

def customlinear_back_real(input1, input2, input3, dim1_0, dim1_1, dim2_0, dim2_1, dim3;_0, device0):
    out = gp_apis.gp_customlinear_back(input1, input2, input3, dim1_0, dim1_1, dim2_0, dim2_1, dim3;_0, device0)
    def grad(dZ1, dZ2, dZ3):
        return gp_apis.gp_customlinear_back(dZ1, dZ2, dZ3, dim1_0, dim1_1, dim2_0, dim2_1, dim3;_0, device0)
    return out, grad

