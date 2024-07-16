Steps to reproduce
==================

Clone repository and fetch the serialized TensorRT engine:

```bash
git clone https://github.com/sergeev917/jetson-nano-jetpack4.6-repro.git
cd jetson-nano-jetpack4.6-repro
git lfs pull
```

Build the test application:

```bash
g++ -O2 -ggdb2 app.cc -o app -I /usr/local/cuda-10.2/include -L /usr/local/cuda-10.2/lib64 -lcuda -lcudart -lnvinfer -lnvonnxparser
```

Run the built application and observe the problem. Note that the application
loads `model.engine` (serialized TensorRT engine) from the working directory,
so the application must be executed from the repository root directory.

```bash
./app
```

The produced output looks like:
```
[MemUsageChange] Init CUDA: CPU +229, GPU +0, now: CPU 311, GPU 2711 (MiB)
Loaded engine size: 33 MiB
Using cublas as a tactic source
[MemUsageChange] Init cuBLAS/cuBLASLt: CPU +159, GPU +163, now: CPU 470, GPU 2875 (MiB)
Using cuDNN as a tactic source
[MemUsageChange] Init cuDNN: CPU +241, GPU +237, now: CPU 711, GPU 3112 (MiB)
Deserialization required 3436767 microseconds.
[MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +33, now: CPU 0, GPU 33 (MiB)
Using cublas as a tactic source
[MemUsageChange] Init cuBLAS/cuBLASLt: CPU +158, GPU +0, now: CPU 710, GPU 3112 (MiB)
Using cuDNN as a tactic source
[MemUsageChange] Init cuDNN: CPU +1, GPU +0, now: CPU 711, GPU 3112 (MiB)
Total per-runner device persistent memory is 8731648
Total per-runner host persistent memory is 13152
Allocated activation device memory of size 344576
[MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +8, now: CPU 0, GPU 41 (MiB)
1: [genericReformat.cu::executeMemcpy::1334] Error Code 1: Cuda Runtime (invalid argument)
app: app.cc:98: int main(): Assertion `ret == cudaSuccess' failed.
Aborted (core dumped)
```

With gdb some internal exceptions can be traced:

```bash
gdb -ex 'catch throw' -ex 'r' ./app
```

The results looks like the following:

```
Catchpoint 1 (throw)
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/aarch64-linux-gnu/libthread_db.so.1".
[New Thread 0x7fa9ef3c80 (LWP 23327)]
[MemUsageChange] Init CUDA: CPU +229, GPU +0, now: CPU 311, GPU 2738 (MiB)
Loaded engine size: 33 MiB
Using cublas as a tactic source
[MemUsageChange] Init cuBLAS/cuBLASLt: CPU +159, GPU +161, now: CPU 470, GPU 2899 (MiB)
Using cuDNN as a tactic source
[MemUsageChange] Init cuDNN: CPU +241, GPU +246, now: CPU 711, GPU 3145 (MiB)
Deserialization required 3485260 microseconds.
[MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +33, now: CPU 0, GPU 33 (MiB)
Using cublas as a tactic source
[MemUsageChange] Init cuBLAS/cuBLASLt: CPU +158, GPU +0, now: CPU 710, GPU 3146 (MiB)
Using cuDNN as a tactic source
[MemUsageChange] Init cuDNN: CPU +1, GPU +0, now: CPU 711, GPU 3146 (MiB)
Total per-runner device persistent memory is 8731648
Total per-runner host persistent memory is 13152
Allocated activation device memory of size 344576
[MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +8, now: CPU 0, GPU 41 (MiB)

Thread 1 "app" hit Catchpoint 1 (exception thrown), 0x0000007fadea0f20 in __cxa_throw () from /usr/lib/aarch64-linux-gnu/libstdc++.so.6
(gdb) bt
#0  0x0000007fadea0f20 in __cxa_throw () from /usr/lib/aarch64-linux-gnu/libstdc++.so.6
#1  0x0000007fae7ce82c in nvinfer1::Lobber<nvinfer1::CudaRuntimeError>::operator()(char const*, char const*, int, int, nvinfer1::ErrorCode, char const*) () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#2  0x0000007faebbdbc8 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#3  0x0000007faebbdc28 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#4  0x0000007faebbdc98 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#5  0x0000007faed55c1c in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#6  0x0000007faeda2720 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#7  0x0000007faeda38e0 in ?? () from /usr/lib/aarch64-linux-gnu/libnvinfer.so.8
#8  0x000000555555656c in nvinfer1::IExecutionContext::executeV2 (this=<optimized out>, bindings=0x7fffffec28) at /usr/include/aarch64-linux-gnu/NvInferRuntime.h:2275
(gdb) c
Continuing.
1: [genericReformat.cu::executeMemcpy::1334] Error Code 1: Cuda Runtime (invalid argument)
app: app.cc:98: int main(): Assertion `ret == cudaSuccess' failed.

Thread 1 "app" received signal SIGABRT, Aborted.
```

While `dmesg` shows the following:

```
....
[418034.716376] nvgpu: 57000000.gpu gk20a_fifo_handle_mmu_fault_locked:1723 [ERR]   mmu fault on engine 0, engine subid 0 (gpc), client 4 (t1 1), addr 0x7fadf9d000, type 3 (va limit viol), access_type 0x00000001,inst_ptr 0x7fe431000
[418034.738271] nvgpu: 57000000.gpu  gk20a_fifo_set_ctx_mmu_error_tsg:1543 [ERR]  TSG 4 generated a mmu fault
[418034.747952] nvgpu: 57000000.gpu   gk20a_fifo_set_ctx_mmu_error_ch:1532 [ERR]  channel 507 generated a mmu fault
[418034.758141] nvgpu: 57000000.gpu   nvgpu_set_error_notifier_locked:137  [ERR]  error notifier set to 31 for ch 507
[418034.768554] nvgpu: 57000000.gpu   gk20a_fifo_set_ctx_mmu_error_ch:1532 [ERR]  channel 506 generated a mmu fault
[418034.778761] nvgpu: 57000000.gpu   nvgpu_set_error_notifier_locked:137  [ERR]  error notifier set to 31 for ch 506
[418034.789151] nvgpu: 57000000.gpu   gk20a_fifo_set_ctx_mmu_error_ch:1532 [ERR]  channel 505 generated a mmu fault
[418034.799370] nvgpu: 57000000.gpu   nvgpu_set_error_notifier_locked:137  [ERR]  error notifier set to 31 for ch 505
[418034.809782] nvgpu: 57000000.gpu   gk20a_fifo_set_ctx_mmu_error_ch:1532 [ERR]  channel 504 generated a mmu fault
[418034.820014] nvgpu: 57000000.gpu   nvgpu_set_error_notifier_locked:137  [ERR]  error notifier set to 31 for ch 504
```

Versions of packages:

```
# dpkg -l | grep -E 'nvidia|infer|cuda'
ii  cuda-command-line-tools-10-2           10.2.460-1                                 arm64        CUDA command-line tools
ii  cuda-compiler-10-2                     10.2.460-1                                 arm64        CUDA compiler
ii  cuda-cudart-10-2                       10.2.300-1                                 arm64        CUDA Runtime native Libraries
ii  cuda-cudart-dev-10-2                   10.2.300-1                                 arm64        CUDA Runtime native dev links, headers
ii  cuda-cuobjdump-10-2                    10.2.300-1                                 arm64        CUDA cuobjdump
ii  cuda-cupti-10-2                        10.2.300-1                                 arm64        CUDA profiling tools runtime libs.
ii  cuda-cupti-dev-10-2                    10.2.300-1                                 arm64        CUDA profiling tools interface.
ii  cuda-documentation-10-2                10.2.300-1                                 arm64        CUDA documentation
ii  cuda-driver-dev-10-2                   10.2.300-1                                 arm64        CUDA Driver native dev stub library
ii  cuda-gdb-10-2                          10.2.300-1                                 arm64        CUDA-GDB
ii  cuda-libraries-10-2                    10.2.460-1                                 arm64        CUDA Libraries 10.2 meta-package
ii  cuda-libraries-dev-10-2                10.2.460-1                                 arm64        CUDA Libraries 10.2 development meta-package
ii  cuda-memcheck-10-2                     10.2.300-1                                 arm64        CUDA-MEMCHECK
ii  cuda-nvcc-10-2                         10.2.300-1                                 arm64        CUDA nvcc
ii  cuda-nvdisasm-10-2                     10.2.300-1                                 arm64        CUDA disassembler
ii  cuda-nvgraph-10-2                      10.2.300-1                                 arm64        NVGRAPH native runtime libraries
ii  cuda-nvgraph-dev-10-2                  10.2.300-1                                 arm64        NVGRAPH native dev links, headers
ii  cuda-nvml-dev-10-2                     10.2.300-1                                 arm64        NVML native dev links, headers
ii  cuda-nvprof-10-2                       10.2.300-1                                 arm64        CUDA Profiler tools
ii  cuda-nvprune-10-2                      10.2.300-1                                 arm64        CUDA nvprune
ii  cuda-nvrtc-10-2                        10.2.300-1                                 arm64        NVRTC native runtime libraries
ii  cuda-nvrtc-dev-10-2                    10.2.300-1                                 arm64        NVRTC native dev links, headers
ii  cuda-nvtx-10-2                         10.2.300-1                                 arm64        NVIDIA Tools Extension
ii  cuda-repo-l4t-10-2-local               10.2.460-1                                 arm64        cuda repository configuration files
ii  cuda-samples-10-2                      10.2.300-1                                 arm64        CUDA example applications
ii  cuda-toolkit-10-2                      10.2.460-1                                 arm64        CUDA Toolkit 10.2 meta-package
ii  cuda-tools-10-2                        10.2.460-1                                 arm64        CUDA Tools meta-package
ii  cuda-visual-tools-10-2                 10.2.460-1                                 arm64        CUDA visual tools
ii  graphsurgeon-tf                        8.2.1-1+cuda10.2                           arm64        GraphSurgeon for TensorRT package
ii  libcudnn8                              8.2.1.32-1+cuda10.2                        arm64        cuDNN runtime libraries
ii  libcudnn8-dev                          8.2.1.32-1+cuda10.2                        arm64        cuDNN development libraries and headers
ii  libcudnn8-samples                      8.2.1.32-1+cuda10.2                        arm64        cuDNN documents and samples
ii  libnvidia-container-tools              1.7.0-1                                    arm64        NVIDIA container runtime library (command-line tools)
ii  libnvidia-container0:arm64             0.10.0+jetpack                             arm64        NVIDIA container runtime library
ii  libnvidia-container1:arm64             1.7.0-1                                    arm64        NVIDIA container runtime library
ii  libnvinfer-bin                         8.2.1-1+cuda10.2                           arm64        TensorRT binaries
ii  libnvinfer-dev                         8.2.1-1+cuda10.2                           arm64        TensorRT development libraries and headers
ii  libnvinfer-doc                         8.2.1-1+cuda10.2                           all          TensorRT documentation
ii  libnvinfer-plugin-dev                  8.2.1-1+cuda10.2                           arm64        TensorRT plugin libraries
ii  libnvinfer-plugin8                     8.2.1-1+cuda10.2                           arm64        TensorRT plugin libraries
ii  libnvinfer-samples                     8.2.1-1+cuda10.2                           all          TensorRT samples
ii  libnvinfer8                            8.2.1-1+cuda10.2                           arm64        TensorRT runtime libraries
ii  libnvonnxparsers-dev                   8.2.1-1+cuda10.2                           arm64        TensorRT ONNX libraries
ii  libnvonnxparsers8                      8.2.1-1+cuda10.2                           arm64        TensorRT ONNX libraries
ii  libnvparsers-dev                       8.2.1-1+cuda10.2                           arm64        TensorRT parsers libraries
ii  libnvparsers8                          8.2.1-1+cuda10.2                           arm64        TensorRT parsers libraries
ii  nvidia-container-csv-cuda              10.2.460-1                                 arm64        Jetpack CUDA CSV file
ii  nvidia-container-csv-cudnn             8.2.1.32-1+cuda10.2                        arm64        Jetpack CUDNN CSV file
ii  nvidia-container-csv-visionworks       1.6.0.501                                  arm64        Jetpack VisionWorks CSV file
ii  nvidia-container-runtime               3.7.0-1                                    all          NVIDIA container runtime
ii  nvidia-container-toolkit               1.7.0-1                                    arm64        NVIDIA container runtime hook
ii  nvidia-docker2                         2.8.0-1                                    all          nvidia-docker CLI wrapper
ii  nvidia-l4t-3d-core                     32.7.4-20230608212426                      arm64        NVIDIA GL EGL Package
ii  nvidia-l4t-apt-source                  32.7.4-20230608212426                      arm64        NVIDIA L4T apt source list debian package
ii  nvidia-l4t-bootloader                  32.7.4-20230608212426                      arm64        NVIDIA Bootloader Package
ii  nvidia-l4t-camera                      32.7.4-20230608212426                      arm64        NVIDIA Camera Package
ii  nvidia-l4t-configs                     32.7.4-20230608212426                      arm64        NVIDIA configs debian package
ii  nvidia-l4t-core                        32.7.4-20230608212426                      arm64        NVIDIA Core Package
ii  nvidia-l4t-cuda                        32.7.4-20230608212426                      arm64        NVIDIA CUDA Package
ii  nvidia-l4t-firmware                    32.7.4-20230608212426                      arm64        NVIDIA Firmware Package
ii  nvidia-l4t-gputools                    32.7.4-20230608212426                      arm64        NVIDIA dgpu helper Package
ii  nvidia-l4t-graphics-demos              32.7.4-20230608212426                      arm64        NVIDIA graphics demo applications
ii  nvidia-l4t-gstreamer                   32.7.4-20230608212426                      arm64        NVIDIA GST Application files
ii  nvidia-l4t-init                        32.7.4-20230608212426                      arm64        NVIDIA Init debian package
ii  nvidia-l4t-initrd                      32.7.4-20230608212426                      arm64        NVIDIA initrd debian package
ii  nvidia-l4t-jetson-io                   32.7.4-20230608212426                      arm64        NVIDIA Jetson.IO debian package
ii  nvidia-l4t-jetson-multimedia-api       32.7.4-20230608212426                      arm64        NVIDIA Jetson Multimedia API is a collection of lower-level APIs that support flexible application development.
ii  nvidia-l4t-kernel                      4.9.337-tegra-32.7.4-20230608212426        arm64        NVIDIA Kernel Package
ii  nvidia-l4t-kernel-dtbs                 4.9.337-tegra-32.7.4-20230608212426        arm64        NVIDIA Kernel DTB Package
ii  nvidia-l4t-kernel-headers              4.9.337-tegra-32.7.4-20230608212426        arm64        NVIDIA Linux Tegra Kernel Headers Package
ii  nvidia-l4t-libvulkan                   32.7.4-20230608212426                      arm64        NVIDIA Vulkan Loader Package
ii  nvidia-l4t-multimedia                  32.7.4-20230608212426                      arm64        NVIDIA Multimedia Package
ii  nvidia-l4t-multimedia-utils            32.7.4-20230608212426                      arm64        NVIDIA Multimedia Package
ii  nvidia-l4t-oem-config                  32.7.4-20230608212426                      arm64        NVIDIA OEM-Config Package
ii  nvidia-l4t-tools                       32.7.4-20230608212426                      arm64        NVIDIA Public Test Tools Package
ii  nvidia-l4t-wayland                     32.7.4-20230608212426                      arm64        NVIDIA Wayland Package
ii  nvidia-l4t-weston                      32.7.4-20230608212426                      arm64        NVIDIA Weston Package
ii  nvidia-l4t-x11                         32.7.4-20230608212426                      arm64        NVIDIA X11 Package
ii  nvidia-l4t-xusb-firmware               32.7.4-20230608212426                      arm64        NVIDIA USB Firmware Package
ii  python3-libnvinfer                     8.2.1-1+cuda10.2                           arm64        Python 3 bindings for TensorRT
ii  python3-libnvinfer-dev                 8.2.1-1+cuda10.2                           arm64        Python 3 development package for TensorRT
ii  tensorrt                               8.2.1.9-1+cuda10.2                         arm64        Meta package of TensorRT
ii  uff-converter-tf                       8.2.1-1+cuda10.2                           arm64        UFF converter for TensorRT package
```
