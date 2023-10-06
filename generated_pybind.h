inline void export_kernel(py::module &m) { 
    m.def("customlinear",[](py::capsule& input1, py::capsule& input2, py::capsule& input3, py::capsule& output1){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> input2_array = capsule_to_array2d(input2);
        array1d_t<float> input3_array = capsule_to_array1d(input3);
        array2d_t<float> output1_array = capsule_to_array2d(output1);
    return customlinear(input1_array, input2_array, input3_array, output1_array);
    }
  );
    m.def("customlinear_back",[](py::capsule& input1, py::capsule& input2, py::capsule& input3, py::capsule& output1, py::capsule& output2, py::capsule& output3){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> input2_array = capsule_to_array2d(input2);
        array2d_t<float> input3_array = capsule_to_array2d(input3);
        array2d_t<float> output1_array = capsule_to_array2d(output1);
        array2d_t<float> output2_array = capsule_to_array2d(output2);
        array1d_t<float> output3_array = capsule_to_array1d(output3);
    return customlinear_back(input1_array, input2_array, input3_array, output1_array, output2_array, output3_array);
    }
  );
}