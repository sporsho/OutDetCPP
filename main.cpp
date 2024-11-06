#include <torch/torch.h>
#include <iostream>
#include "src/outdet.h"
#include<string>
#include<iomanip>
#include<vector>
#include<fstream>
#include<tuple>
#include<chrono>
#include<cuda_runtime.h>
#include<helper_functions.h>
#include<helper_cuda.h>
#include<dlfcn.h>
#include<filesystem>
#include<stdlib.h>
#include <assert.h>
#include <filesystem>
#include "src/knncuda.h"
#include <torch/script.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include<cuda_runtime.h>
//#include <rmm/cuda_stream_view.hpp>
//#include <cuml/common/logger.hpp>
//#include<cuml/neighbors/knn.hpp>
using namespace nvinfer1;
using namespace nvonnxparser;
class Logger : public ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        if (severity <= Severity::kWARNING){
            std::cout << msg << std::endl;
        }
    }
} logger;
using namespace torch::indexing;
using namespace std::filesystem;
bool load_cpp_weights(OutDet model){
    // load model weights
    torch::jit::script::Module cpp_module = torch::jit::load("/var/local/home/aburai/outdet_cpp/cpp_weights.pt");
    // nh conv layer 1
    assert(cpp_module.hasattr("convs.0.conv1.dw") == true && "Cannot read convs.0.conv1.dw");
    model->conv1->conv1->dw = cpp_module.attr("convs.0.conv1.dw").toTensor();
//    std::cout << model->conv1->conv1->dw << std::endl;
assert(cpp_module.hasattr("convs.0.conv1.b") == true && "Cannot read convs.0.conv1.b");
    model->conv1->conv1->b = cpp_module.attr("convs.0.conv1.b").toTensor();
    assert(cpp_module.hasattr("convs.0.conv1.weight") == true && "Cannot read convs.0.conv1.weight");
    model->conv1->conv1->weight = cpp_module.attr("convs.0.conv1.weight").toTensor();
    // batch norm 1
    assert(cpp_module.hasattr("convs.0.bn1.weight") == true && "Cannot read convs.0.bn1.weight");
    model->conv1->bn1->weight = cpp_module.attr("convs.0.bn1.weight").toTensor();
    assert(cpp_module.hasattr("convs.0.bn1.bias") == true && "Cannot read convs.0.bn1.bias");
    model->conv1->bn1->bias = cpp_module.attr("convs.0.bn1.bias").toTensor();
    assert(cpp_module.hasattr("convs.0.bn1.running_mean") == true && "Cannot read convs.0.bn1.running_mean");
    model->conv1->bn1->running_mean = cpp_module.attr("convs.0.bn1.running_mean").toTensor();
    assert(cpp_module.hasattr("convs.0.bn1.running_var") == true && "Cannot read convs.0.bn1.running_var");
    model->conv1->bn1->running_var = cpp_module.attr("convs.0.bn1.running_var").toTensor();
    assert(cpp_module.hasattr("convs.0.bn1.num_batches_tracked") == true && "Cannot read convs.0.bn1.num_batches_tracked");
    model->conv1->bn1->num_batches_tracked = cpp_module.attr("convs.0.bn1.num_batches_tracked").toTensor();

    // nh conv layer 2
    assert(cpp_module.hasattr("convs.0.conv2.weight") == true && "Cannot read convs.0.conv2.weight");
    model->conv1->conv2->weight = cpp_module.attr("convs.0.conv2.weight").toTensor();

    //batch norm 2
    assert(cpp_module.hasattr("convs.0.bn2.weight") == true && "Cannot read convs.0.bn2.weight");
    model->conv1->bn2->weight = cpp_module.attr("convs.0.bn2.weight").toTensor();
    assert(cpp_module.hasattr("convs.0.bn2.bias") == true && "Cannot read convs.0.bn2.bias");
    model->conv1->bn2->bias = cpp_module.attr("convs.0.bn2.bias").toTensor();
    assert(cpp_module.hasattr("convs.0.bn2.running_mean") == true && "Cannot read convs.0.bn2.running_mean");
    model->conv1->bn2->running_mean = cpp_module.attr("convs.0.bn2.running_mean").toTensor();
    assert(cpp_module.hasattr("convs.0.bn2.running_var") == true && "Cannot read convs.0.bn2.running_var");
    model->conv1->bn2->running_var = cpp_module.attr("convs.0.bn2.running_var").toTensor();
    assert(cpp_module.hasattr("convs.0.bn2.num_batches_tracked") == true && "Cannot read convs.0.bn2.num_batches_tracked");
    model->conv1->bn2->num_batches_tracked = cpp_module.attr("convs.0.bn2.num_batches_tracked").toTensor();

    // residual connection
    assert(cpp_module.hasattr("convs.0.downsample.weight") == true && "Cannot read convs.0.downsample.weight");
    model->conv1->downsample->weight = cpp_module.attr("convs.0.downsample.weight").toTensor();

    // linear layer
    assert(cpp_module.hasattr("fc.weight") == true && "Cannot read fc.weight");
    model->fc->weight = cpp_module.attr("fc.weight").toTensor();
    assert(cpp_module.hasattr("fc.bias") == true && "Cannot read fc.bias");
    model->fc->bias = cpp_module.attr("fc.bias").toTensor();

return true;
}

struct InferDeleter{
    template<typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};
template <typename T>
using TRTUniquePointer = std::unique_ptr<T, InferDeleter>;
int main(int argc, char **argv)
{
    try{
        // set device
        torch::Device device(torch::kCUDA, 0);
        // single point cloud
        const char* fname =  "/var/local/home/aburai/DATA/WADS2/sequences/11/velodyne/039498.bin";
        std::ifstream fin(fname, std::ios::binary);
        fin.seekg(0, std::ios::end);
        const size_t num_elements = fin.tellg() / sizeof(float);
        fin.seekg(0, std::ios::beg);
        float data[num_elements];
        fin.read(reinterpret_cast<char*>(&data[0]), num_elements * sizeof(float));
        // initial sizes
        long a_init = long(num_elements / 4);
        long b = 4;
        auto mean = torch::tensor({0.3420934,  -0.01516175 ,-0.5889243 ,  9.875928}, torch::kFloat32).unsqueeze(0);
        auto stddev = torch::tensor({25.845459,  18.93466,    1.5863657, 14.734034}, torch::kFloat32).unsqueeze(0);

//        std::cout << mean << std::endl;

        auto pt_tensor = torch::from_blob(data, {a_init, 4}, torch::TensorOptions().dtype(torch::kFloat));
        torch::Tensor pt_xyz = pt_tensor.index({Slice(), Slice(None, 3)});
//
        auto o_dist = torch::pow(pt_xyz, 2).sum(1);
//        std::cout << o_dist.sizes() << std::endl;
        auto selected_ind = torch::where(o_dist < 10 * 10)[0];
//        auto o_dist = pt
//        std::cout << selected_ind<< std::endl;
        auto selected_data = pt_tensor.index({selected_ind});
        // sizes to be used
        float *selected_data_arr = selected_data.data_ptr<float>();
        long num_feat = 3;
        int a = selected_data.size(0);  // the code breaks at 40k points
        int BatchSize = a;
        mean = mean.repeat({ a, 1});
        stddev = stddev.repeat({a, 1});
        // data holder for points
        float points[a * num_feat];
        // copy points
        for (int i =0; i < a; i++){
            for (int j =0; j < 3; j ++){
                points[num_feat * i + j] = selected_data_arr[b * i + j];
            }
        }

//        float cat_arr[9][2] = {{0,0}, {1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6}, {7,7}, {8,8}};
        float *ref = (float*)malloc(a * num_feat * sizeof(float ));
        int k = 9;
        memcpy(ref, &points, sizeof(float) * a * num_feat);


        float *query = (float *)malloc(sizeof(float) * a * num_feat);
        memcpy(query, &points, sizeof(float) * a * num_feat);

        float *knn_dist = (float*)malloc(a *a *  sizeof(float ));

//        std::cout << a_init << std::endl;

        kNN_dist(ref, a, query, a, num_feat, knn_dist);
//        print_array(knn_dist, a, a, a);
        auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor dist_tensor = torch::from_blob(knn_dist, {a, a}, options);
        dist_tensor = dist_tensor.to(device);
        auto [dist, ind] = dist_tensor.topk(9, 1, false, true);
//        std::cout << dist.sizes()<< " " << ind.sizes() << std::endl;

        auto inp = selected_data.sub(mean).div(stddev);
//        torch::Tensor inp = pt_tensor.index({Slice(None, a)});
        dist = torch::sqrt(dist) + 1.0;
//        std::cout << inp << std::endl;
        // define model
        OutDet model(2, 1, 3, 4, 32);
        model->to(device);
        load_cpp_weights(model);
        torch::NoGradGuard no_grad;
        model->eval();
//        auto out = model->forward(inp.to(device), dist.to(device), ind.to(device));
//        out = out.argmax(1);
//        out = out.contiguous();
//        out = out.to(torch::kCPU);
//        long int * pred = out.data_ptr< long int>();
//         int counter = 0;
//        for (int i = 0; i < out.numel(); i++){
//            if (pred[i] == 1){
//                counter++;
//            }
//        }
////        std::cout << out.numel() << std::endl;
//        std::cout << counter << std::endl;
        //        std::cout << out << std::endl;
//        torch::save(model->parameters(), "mymodel.pt");
//        std::cout << model->conv1->conv1->dw << std::endl;

////        std::cout << model->conv1->conv1->dw << std::endl;
//        if ( !is_directory("../saved_weights") || !exists("../saved_weights")){
//            create_directory("../saved_weights");
//        }
//        torch::save(model, "../saved_weights/outdet.pt");
//            torch::load(model, "../saved_weights/outdet.pt");
//        std::cout << model->conv1->conv1->dw << std::endl;
        IBuilder* builder = createInferBuilder(logger);
        INetworkDefinition* network = builder->createNetworkV2(0);
        IParser* parser = createParser(*network, logger);
        std::ifstream file("/var/local/home/aburai/outdet_cpp/outdet.onnx", std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (! file.read(buffer.data(), size)){
            std::cout << "Error"<< std::endl;
            throw std::runtime_error("Cannot read file");
        }
        auto success = parser->parse(buffer.data(), buffer.size());
        if (!success){
            throw std::runtime_error("failed to parse model");
        }
        const auto numInputs  = network->getNbInputs();
        std::cout << "Number of inputs: " << numInputs << std::endl;
        const auto input0batch = network->getInput(0)->getDimensions().d[0];
        std::cout << "Batch Size: " << input0batch << std::endl;
        // create build config
        auto conf_succes = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
        if (!conf_succes){
            throw std::runtime_error("Cannot create build config");
        }
        // register single optimization profile
        IOptimizationProfile* optProfile = builder->createOptimizationProfile();
        for (int32_t i =0; i < numInputs; i++){
            const auto input = network->getInput(i);
            const auto inputName = input->getName();
            const auto inputDims = input->getDimensions();
            int32_t inputF = inputDims.d[1];
            optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims2(1, inputF));
            optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims2(250000, inputF));
            optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims2(30000, inputF));
        }
        conf_succes->addOptimizationProfile(optProfile);
        // use default precision
        cudaStream_t profileStream;
        cudaStreamCreate(&profileStream);
        conf_succes->setProfileStream(profileStream);

//        IBuilderConfig* config = builder->createBuilderConfig();
        conf_succes->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1024*1024*1024);
        conf_succes->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 1024*1024*1024);
        std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *conf_succes)};
//        IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *conf_succes);

        // write engine to disk
        const auto enginePath = "/var/local/home/aburai/outdet_cpp/outdet.trt";
        std::ofstream outfile(enginePath, std::ofstream::binary);
        outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());

        std::shared_ptr<IRuntime> mRuntime;
        std::shared_ptr<ICudaEngine> mEngine;
        mRuntime = std::shared_ptr<IRuntime>(createInferRuntime(logger));
        mEngine = std::shared_ptr<ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
        // create context
//        std::shared_ptr<IExecutionContext> context
        TRTUniquePointer<IExecutionContext> context(mEngine->createExecutionContext(), InferDeleter());
        // create buffer object
        std::vector<void *> m_buffers;
        std::vector<int32_t > m_outputLengths{};
        std::vector<int32_t> m_inputDims;
        std::vector<int32_t> m_outputDims;
        std::vector<std::string> m_IOTensorNames;
//        BufferManager buffer(mEngine);
        cudaStream_t stream;
        m_buffers.resize(mEngine->getNbIOTensors());
        auto err = cudaStreamCreate(&stream);
        // first input
        int32_t m_inputBatchSize;
        int32_t  maxBatchSize =250000;
        Dims2 inp1Dims = {BatchSize, 4};
        for (int i=0; i< mEngine->getNbIOTensors(); i++){
            const auto tensorName = mEngine->getIOTensorName(i);
//            std::cout << tensorName << std::endl;
            m_IOTensorNames.emplace_back(tensorName);
            const auto tensorType = mEngine->getTensorIOMode(tensorName);
            const auto tensorShape = mEngine->getTensorShape(tensorName);
            const auto tensorDataType = mEngine->getTensorDataType(tensorName);

            if (tensorType == TensorIOMode::kINPUT){
                m_inputDims.emplace_back(tensorShape.d[1]);
                m_inputBatchSize = tensorShape.d[0];
                std::cout << tensorShape.d[1] << std::endl;

            }
            else if (tensorType == TensorIOMode::kOUTPUT){
                if (tensorDataType == DataType::kFLOAT){
                    std::cout << "float out " <<tensorShape.d[1] << std::endl;
                    m_outputDims.push_back(tensorShape.d[1]);
                    m_outputLengths.push_back(tensorShape.d[1]);
                    auto err = cudaMallocAsync(&m_buffers[i], tensorShape.d[1] * maxBatchSize * sizeof(float), stream);
                    if (err != cudaSuccess){
                        std::cout << err << std::endl;
                    }
                }
            }
        }


        // destroy cuda steam at the end
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
//        const auto batchSize = -1;
        Dims2 inp1Shape = Dims2(BatchSize, 4);
        Dims2 inp2Shape = Dims2(BatchSize, 9);
        Dims2 inp3Shape = Dims2(BatchSize, 9);
        context->setInputShape("points", inp1Dims);
        context->setInputShape("dist", inp2Shape);
        context->setInputShape("indices", inp3Shape);
        m_buffers[0] = (void *)inp.to(device).data_ptr();
        m_buffers[1] = (void *)dist.to(device).data_ptr();
        m_buffers[2] = (void *)ind.to(device).data_ptr();
        // set the address of input and output buffer
        context->setTensorAddress("points", m_buffers[0]);
        context->setTensorAddress("dist", m_buffers[1]);
        context->setTensorAddress("indices", m_buffers[2]);
        context->setTensorAddress("out", m_buffers[3]);
        // infer
        auto status = context->enqueueV3(profileStream);
        float *rt_out = (float *)malloc(sizeof(float) * BatchSize * 2);
        cudaMemcpyAsync(rt_out, m_buffers[3], sizeof(float) * BatchSize * 2, cudaMemcpyDeviceToHost);
        torch::Tensor tensor_out = torch::from_blob(rt_out, {BatchSize, 2}, torch::TensorOptions().dtype(torch::kFloat));
        cudaStreamDestroy(profileStream);

        auto out = tensor_out.argmax(1);
        out = out.contiguous();
        out = out.to(torch::kCPU);
        long int * pred = out.data_ptr< long int>();
         int counter = 0;
        for (int i = 0; i < out.numel(); i++){
            if (pred[i] == 1){
                counter++;
            }
        }
        std::cout << counter << std::endl;



    }
    catch (const std::exception &ex){
        std::cerr << ex.what() << std::endl;
    }
    return 0;
}
