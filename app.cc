#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

using std::cerr;
using std::cout;
using std::ifstream;
using std::ios;
using std::istreambuf_iterator;
using std::move;
using std::size_t;
using std::unique_ptr;
using std::vector;
using nvinfer1::BuilderFlag;
using nvinfer1::IBuilder;
using nvinfer1::IBuilderConfig;
using nvinfer1::ICudaEngine;
using nvinfer1::IExecutionContext;
using nvinfer1::IHostMemory;
using nvinfer1::ILogger;
using nvinfer1::INetworkDefinition;
using nvinfer1::IRuntime;
using nvinfer1::ITimingCache;
using nvinfer1::NetworkDefinitionCreationFlag;
using nvinfer1::createInferBuilder;
using nvinfer1::createInferRuntime;
using nvonnxparser::IParser;
using nvonnxparser::createParser;

#define check_call(expr) { const cudaError_t ret = (expr); assert(ret == cudaSuccess); }

struct InferLogger: ILogger {
    void log(Severity, const char * msg) noexcept override { cerr << msg << '\n'; }
};

int main()
{
    constexpr size_t input_size = 1 * 3 * 56 * 128 * sizeof(float);
    constexpr size_t output_size = 2 * 2 * sizeof(float);

    vector<char> engine_buf;
    {
        ifstream f("model.engine", ios::binary);
        assert(bool(f));
        engine_buf = vector<char>(
            istreambuf_iterator<char>{f},
            istreambuf_iterator<char>{}
        );
    }

    InferLogger logger;
    auto runtime_ptr = unique_ptr<IRuntime>{createInferRuntime(logger)};
    assert(runtime_ptr != nullptr);

    auto engine_ptr = unique_ptr<ICudaEngine>{
        runtime_ptr->deserializeCudaEngine(
            engine_buf.data(),
            engine_buf.size()
        )
    };
    assert(engine_ptr != nullptr);

    auto exec_ctx_ptr = unique_ptr<IExecutionContext>{
        engine_ptr->createExecutionContext()
    };
    assert(exec_ctx_ptr != nullptr);
    assert(exec_ctx_ptr->allInputDimensionsSpecified());

    void * input_dev_ptr = nullptr;
    check_call(cudaMalloc(&input_dev_ptr, input_size));
    assert(input_dev_ptr != nullptr);

    void * output_dev_ptr = nullptr;
    check_call(cudaMalloc(&output_dev_ptr, output_size));
    assert(output_dev_ptr != nullptr);

    const int input_idx = engine_ptr->getBindingIndex("data");
    const int output_idx = engine_ptr->getBindingIndex("out");
    assert(input_idx == 0 || input_idx == 1);
    assert(output_idx == 0 || output_idx == 1);
    assert(input_idx != output_idx);
    void * bound_ptrs[2]{};
    bound_ptrs[input_idx] = input_dev_ptr;
    bound_ptrs[output_idx] = output_dev_ptr;

    exec_ctx_ptr->executeV2(&bound_ptrs[0]);

    /* cleanup */
    check_call(cudaFree(output_dev_ptr));
    check_call(cudaFree(input_dev_ptr));
    exec_ctx_ptr.reset();
    engine_ptr.reset();
    runtime_ptr.reset();
    engine_buf.clear();
    return 0;
}
