#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
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
using std::string;
using std::unique_ptr;
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
#define check_state() { const cudaError_t ret = cudaPeekAtLastError(); assert(ret == cudaSuccess); }

struct InferLogger: ILogger {
    void log(Severity severity, const char * msg) noexcept override
    {
        cerr << msg << '\n';
    }
};

int main()
{
    constexpr size_t input_size = 1 * 3 * 56 * 128 * sizeof(float);
    constexpr size_t output_size = 2 * 2 * sizeof(float);

    string onnx_buf;
    {
        ifstream f("model.onnx", ios::binary);
        assert(bool(f));
        onnx_buf.assign(
            istreambuf_iterator<char>{f},
            istreambuf_iterator<char>{}
        );
    }

    InferLogger logger;
    auto builder_ptr = unique_ptr<IBuilder>{createInferBuilder(logger)};
    assert(builder_ptr != nullptr);
    builder_ptr->setMaxBatchSize(1u);
    check_state();

    auto cfg_ptr = unique_ptr<IBuilderConfig>{builder_ptr->createBuilderConfig()};
    assert(cfg_ptr != nullptr);
    cfg_ptr->setMaxWorkspaceSize(256 * 1024 * 1024);
    if (builder_ptr->platformHasFastFp16()) {
        cfg_ptr->setFlag(BuilderFlag::kFP16);
    }
    check_state();

    auto def_ptr = unique_ptr<INetworkDefinition>{
        builder_ptr->createNetworkV2(
            1U << static_cast<uint32_t>(
                NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
            )
        )
    };
    assert(def_ptr != nullptr);
    check_state();

    auto parser_ptr = unique_ptr<IParser>{createParser(*def_ptr, logger)};
    assert(parser_ptr != nullptr);
    bool parse_ok = parser_ptr->parse(onnx_buf.data(), onnx_buf.size());
    assert(parse_ok);
    check_state();

    auto serialized_engine = unique_ptr<IHostMemory>{
        builder_ptr->buildSerializedNetwork(*def_ptr, *cfg_ptr)
    };
    assert(serialized_engine != nullptr);
    check_state();

    auto runtime_ptr = unique_ptr<IRuntime>{createInferRuntime(logger)};
    assert(runtime_ptr != nullptr);
    check_state();

    auto engine_ptr = unique_ptr<ICudaEngine>{
        runtime_ptr->deserializeCudaEngine(
            serialized_engine->data(),
            serialized_engine->size()
        )
    };
    assert(engine_ptr != nullptr);
    check_state();

    auto exec_ctx_ptr = unique_ptr<IExecutionContext>{
        engine_ptr->createExecutionContext()
    };
    assert(exec_ctx_ptr != nullptr);
    assert(exec_ctx_ptr->allInputDimensionsSpecified());
    check_state();

    void * input_dev_ptr = nullptr;
    check_call(cudaMalloc(&input_dev_ptr, input_size));
    assert(input_dev_ptr != nullptr);
    check_call(cudaMemset(input_dev_ptr, 0, input_size));
    check_state();

    void * output_dev_ptr = nullptr;
    check_call(cudaMalloc(&output_dev_ptr, output_size));
    assert(output_dev_ptr != nullptr);
    check_state();

    void * output_host_ptr = nullptr;
    check_call(cudaMallocHost(&output_host_ptr, output_size));
    assert(output_host_ptr != nullptr);
    check_state();

    const int input_idx = engine_ptr->getBindingIndex("data");
    const int output_idx = engine_ptr->getBindingIndex("out");
    assert(input_idx == 0 || input_idx == 1);
    assert(output_idx == 0 || output_idx == 1);
    assert(input_idx != output_idx);
    void * bound_ptrs[2]{};
    bound_ptrs[input_idx] = input_dev_ptr;
    bound_ptrs[output_idx] = output_dev_ptr;

    exec_ctx_ptr->executeV2(&bound_ptrs[0]);
    check_state();

    check_call(
        cudaMemcpy(
            output_host_ptr,
            output_dev_ptr,
            output_size,
            cudaMemcpyDeviceToHost
        )
    );

    // no cleanup, it fails earlier anyway
    cout << "Looks OK!\n";

    return 0;
}
