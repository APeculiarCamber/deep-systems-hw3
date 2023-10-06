#pragma once
#include "csr.h"
#include "op.h"

void customlinear(array2d_t<float>& input1, array2d_t<float>& input2, array1d_t<float>& input3, array2d_t<float>& output1);
void customlinear_back(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& input3, array2d_t<float>& output1, array2d_t<float>& output2, array1d_t<float>& output3);