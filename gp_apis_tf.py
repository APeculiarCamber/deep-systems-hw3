import tensorlow as tf
import kernel as gpk
def gp_customlinear(X1, X2, X3, dim1;_0, dim1;_1):
    X1_dl = tf.experimental.dlpack.to_dlpack(X1)
    X2_dl = tf.experimental.dlpack.to_dlpack(X2)
    X3_dl = tf.experimental.dlpack.to_dlpack(X3)
    #declare the output tensor here
    res = tf.zeros([dim_0, dim_1])
    res_dl = tf.experimental.dlpack.to_dlpack(res)
    gpk.customlinear(X1_dl, X2_dl, X3_dl, res_dl)
    return res
def gp_customlinear_back(X1, X2, X3, dim1_0, dim1_1, dim2_0, dim2_1, dim3;_0):
    X1_dl = tf.experimental.dlpack.to_dlpack(X1)
    X2_dl = tf.experimental.dlpack.to_dlpack(X2)
    X3_dl = tf.experimental.dlpack.to_dlpack(X3)
    #declare the output tensor here
    res1 = tf.zeros([dim1_0, dim1_1])
    res_dl1 = tf.experimental.dlpack.to_dlpack(res1)
    res2 = tf.zeros([dim2_0, dim2_1])
    res_dl2 = tf.experimental.dlpack.to_dlpack(res2)
    res3 = tf.zeros([dim3_0])
    res_dl3 = tf.experimental.dlpack.to_dlpack(res3)
    gpk.customlinear_back(X1_dl, X2_dl, X3_dl, res_dl1, res_dl2, res_dl3)
    return res1, res2, res3
