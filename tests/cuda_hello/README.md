# CUDA vector-add unit test

This example builds a CUDA vector-add kernel and runs GTest cases on a GPU.
It checks results and the output boundary for sizes around a CUDA block, as
well as empty and invalid inputs, using both the default and a custom stream.

Requirements:

- Bazel with Bzlmod support and a C++ compiler.
- CUDA Toolkit at `/usr/local/cuda`, including `nvcc` and `libcudart`.
- A CUDA-capable GPU and driver. The kernel is compiled to compute capability
  7.5 PTX, requiring a GPU with compute capability 7.5 or newer.
- System GTest headers and libraries (on Ubuntu: `libgtest-dev`).

From the repository root:

```bash
bazel build //tests/cuda_hello:cuda_hello
bazel test //tests/cuda_hello:cuda_hello --test_output=errors
```

`bazel test` may reuse a previous result. To actually run the GPU test again,
add `--nocache_test_results`. CUDA and GTest are taken from the local system,
so changing those installations may require `bazel clean` before rebuilding.