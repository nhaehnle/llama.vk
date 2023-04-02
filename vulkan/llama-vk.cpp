
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../llama.h"

#include "vulkan/vulkan.h"

#define LLVK_CHECK_RESULT(result) llvk_check_result(result, #result, __FILE__, __LINE__)

static VkResult llvk_check_result(VkResult result, const char *call, const char *file, unsigned line) {
    if (result < 0) {
        fprintf(stderr, "Vulkan API call failed\n");
        fprintf(stderr, "  Result code: %d\n", result);
        fprintf(stderr, "  Call: %s\n", call);
        fprintf(stderr, "  at %s:%u\n", file, line);
        exit(1);
    }

    return result;
}

namespace llvk {

class Instance {
protected:
    VkInstance theInstance = nullptr;

public:
    Instance() = default;
    explicit Instance(VkInstance instance);

    operator VkInstance() { return theInstance; }

#define FN(name) PFN_vk ## name name = nullptr;
#include "llama-vk-functions.inc"
};

class OwnedInstance : public Instance {
public:
    OwnedInstance() = default;
    ~OwnedInstance();

    explicit OwnedInstance(VkInstance instance) : Instance(instance) {}

    OwnedInstance(OwnedInstance &&rhs) : Instance(rhs) {
        rhs.theInstance = nullptr;
    }
    OwnedInstance &operator=(OwnedInstance &&rhs) {
        if (&rhs != this) {
            if (theInstance)
                DestroyInstance(theInstance, NULL);
            this->Instance::operator=(rhs);
            rhs.theInstance = nullptr;
        }
        return *this;
    }

    static OwnedInstance createDefault();

private:
    OwnedInstance(const OwnedInstance &rhs) = delete;
    OwnedInstance &operator=(const OwnedInstance &rhs) = delete;
};

Instance::Instance(VkInstance instance) : theInstance(instance) {
#define FN(name) name = reinterpret_cast<PFN_vk ## name>(vkGetInstanceProcAddr(theInstance, "vk" # name));
#include "llama-vk-functions.inc"
}

OwnedInstance OwnedInstance::createDefault() {
    PFN_vkEnumerateInstanceLayerProperties EnumerateInstanceLayerProperties = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceLayerProperties"));
    PFN_vkCreateInstance CreateInstance = reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr(NULL, "vkCreateInstance"));

    uint32_t layerCount;
    std::vector<VkLayerProperties> layerProperties;
    LLVK_CHECK_RESULT(EnumerateInstanceLayerProperties(&layerCount, NULL));
    layerProperties.resize(layerCount);
    LLVK_CHECK_RESULT(EnumerateInstanceLayerProperties(&layerCount, layerProperties.data()));
    layerProperties.resize(layerCount);

    bool foundValidationLayer = false;

    printf("vulkan: available layers:\n");
    for (const VkLayerProperties &layer : layerProperties) {
        printf(" - %s (spec: %u implementation: %u)\n", layer.layerName, layer.specVersion,
               layer.implementationVersion);
        printf("   %s\n", layer.description);

        if (!strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation"))
            foundValidationLayer = true;
    }

    printf("vulkan: Found validation layer\n");

    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "llama-vk";
    applicationInfo.applicationVersion = 0;
    applicationInfo.apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);

    static const std::array validationFeatureEnables{
        VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,
        VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
    };
    VkValidationFeaturesEXT validationFeatures = {};
    validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
    validationFeatures.enabledValidationFeatureCount = validationFeatureEnables.size();
    validationFeatures.pEnabledValidationFeatures = validationFeatureEnables.data();

    static const std::array layerNames{
        "VK_LAYER_KHRONOS_validation",
    };
    static const std::array instanceExtensions{
        "VK_EXT_validation_features",
    };

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    // createInfo.pNext = &validationFeatures;
    createInfo.pApplicationInfo = &applicationInfo;
    createInfo.enabledLayerCount = layerNames.size();
    createInfo.ppEnabledLayerNames = layerNames.data();
    createInfo.enabledExtensionCount = instanceExtensions.size();
    createInfo.ppEnabledExtensionNames = instanceExtensions.data();

    VkInstance instance;
    LLVK_CHECK_RESULT(CreateInstance(&createInfo, NULL, &instance));
    return OwnedInstance(instance);
}

OwnedInstance::~OwnedInstance() {
    if (theInstance)
        DestroyInstance(theInstance, NULL);
}

class Device {
protected:
    Instance *vk = nullptr;
    VkDevice device = nullptr;
    VkQueue computeQueue = nullptr;
    VkQueue transferQueue = nullptr;

    Device() = default;

public:
    Device(Instance &vk, VkDevice device, VkQueue computeQueue, VkQueue transferQueue)
        : vk(&vk), device(device), computeQueue(computeQueue)
        , transferQueue(transferQueue) {}

    operator VkDevice() { return device; }
};

class OwnedDevice : public Device {
public:
    OwnedDevice() = default;
    ~OwnedDevice() {
        if (device)
            vk->DestroyDevice(device, NULL);
    }

    OwnedDevice(OwnedDevice &&rhs) : Device(rhs) {
        rhs.device = nullptr;
    }
    OwnedDevice &operator=(OwnedDevice &&rhs) {
        if (&rhs != this) {
            if (device)
                vk->DestroyDevice(device, NULL);
            this->Device::operator=(rhs);
            rhs.device = nullptr;
        }
        return *this;
    }

    static OwnedDevice createDefault(Instance &vk);

private:
    OwnedDevice(const OwnedDevice &rhs) = delete;
    OwnedDevice &operator=(const OwnedDevice &rhs) = delete;

};

static unsigned deviceTypePriority(VkPhysicalDeviceType type) {
    switch (type) {
    default:
    case VK_PHYSICAL_DEVICE_TYPE_OTHER:
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return 0;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
    }
}

OwnedDevice OwnedDevice::createDefault(Instance &vk) {
    uint32_t physicalDeviceCount;
    std::vector<VkPhysicalDevice> physicalDevices;
    LLVK_CHECK_RESULT(vk.EnumeratePhysicalDevices(vk, &physicalDeviceCount, NULL));
    physicalDevices.resize(physicalDeviceCount);
    LLVK_CHECK_RESULT(vk.EnumeratePhysicalDevices(vk, &physicalDeviceCount, physicalDevices.data()));
    physicalDevices.resize(physicalDeviceCount);

    int best = -1;
    VkPhysicalDeviceType bestType = VK_PHYSICAL_DEVICE_TYPE_OTHER;

    printf("vulkan: available physical devices:\n");
    for (unsigned idx = 0; idx < physicalDevices.size(); ++idx) {
        auto physicalDevice = physicalDevices[idx];

        VkPhysicalDeviceVulkan12Properties vulkan12 = {};
        vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;

        VkPhysicalDeviceVulkan11Properties vulkan11 = {};
        vulkan11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
        vulkan11.pNext = &vulkan12;

        VkPhysicalDeviceProperties2 properties = {};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &vulkan11;

        vk.GetPhysicalDeviceProperties2(physicalDevice, &properties);

        VkPhysicalDeviceType deviceType = properties.properties.deviceType;
        const char *deviceTypeStr = "<unknown>";
        switch (deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_OTHER: deviceTypeStr = "other"; break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: deviceTypeStr = "integrated GPU"; break;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: deviceTypeStr = "discrete GPU"; break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: deviceTypeStr = "virtual GPU"; break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: deviceTypeStr = "CPU"; break;
        default: deviceTypeStr = "<unknown>"; break;
        }
        printf(" %u. Device: %s (%s, Vulkan version %u.%u.%u)\n",
               idx, properties.properties.deviceName, deviceTypeStr,
               VK_API_VERSION_MAJOR(properties.properties.apiVersion),
               VK_API_VERSION_MINOR(properties.properties.apiVersion),
               VK_API_VERSION_PATCH(properties.properties.apiVersion));
        printf("   Driver Name: %s\n", vulkan12.driverName);
        printf("   Driver Info: %s\n", vulkan12.driverInfo);

        auto reportMissingFeature = [](const char *name) {
            printf("   --- ignoring device because it does not support %s\n", name);
        };

        if (properties.properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 3, 0)) {
            reportMissingFeature("Vulkan 1.3");
            continue;
        }

        VkPhysicalDeviceVulkan13Features vulkan13Features = {};
        vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

        VkPhysicalDeviceVulkan12Features vulkan12Features = {};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12Features.pNext = &vulkan13Features;

        VkPhysicalDeviceVulkan11Features vulkan11Features = {};
        vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vulkan11Features.pNext = &vulkan12Features;

        VkPhysicalDeviceFeatures2 features = {};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &vulkan11Features;
        vk.GetPhysicalDeviceFeatures2(physicalDevice, &features);

#define FEATURE_CHECK(strct, field) \
        if (!strct.field) { \
            reportMissingFeature(#field); \
            continue; \
        }

        FEATURE_CHECK(features.features, shaderInt16)
        FEATURE_CHECK(vulkan11Features, storageBuffer16BitAccess)
        FEATURE_CHECK(vulkan12Features, shaderFloat16)
        FEATURE_CHECK(vulkan13Features, computeFullSubgroups)

#undef FEATURE_CHECK

        if (best < 0 || deviceTypePriority(deviceType) > deviceTypePriority(bestType)) {
            best = idx;
            bestType = deviceType;
        }
    }

    if (best < 0) {
        fprintf(stderr, "vulkan: no suitable device found!\n");
        exit(1);
    }

    printf("vulkan: choosing device with index %u\n", best);

    VkPhysicalDevice physicalDevice = physicalDevices[best];

    uint32_t queueFamilyCount;
    std::vector<VkQueueFamilyProperties> queueFamilyProperties;
    vk.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);
    queueFamilyProperties.resize(queueFamilyCount);
    vk.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                              queueFamilyProperties.data());
    queueFamilyProperties.resize(queueFamilyCount);

    int computeQueueFamily = -1;
    int transferQueueFamily = -1;

    printf("vulkan: available queue families:\n");
    for (unsigned idx = 0; idx < queueFamilyCount; ++idx) {
        const auto &properties = queueFamilyProperties[idx];

        bool haveGraphics = properties.queueFlags & VK_QUEUE_GRAPHICS_BIT;
        bool haveCompute = properties.queueFlags & VK_QUEUE_COMPUTE_BIT;
        bool haveTransfer = properties.queueFlags & VK_QUEUE_TRANSFER_BIT;

        if (haveCompute) {
            bool pick = computeQueueFamily < 0;

            // Prefer a non-graphics queue family so that we don't affect desktop
            // responsiveness as much if we send long-running command buffers.
            if (!pick && !haveGraphics &&
                (queueFamilyProperties[computeQueueFamily].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                pick = true;
            }

            if (pick)
                computeQueueFamily = idx;
        }

        if (haveTransfer && !haveGraphics && !haveCompute) {
            if (transferQueueFamily < 0)
                transferQueueFamily = idx;
        }

        printf(" %u. Flags: ", idx);

        VkQueueFlags flags = properties.queueFlags;
#define PRINT_FLAG(flag, name) \
        if (flags & (flag)) { \
            if (flags != properties.queueFlags) printf("|"); \
            printf(name); \
            flags &= ~(flag); \
        }
        PRINT_FLAG(VK_QUEUE_GRAPHICS_BIT, "graphics")
        PRINT_FLAG(VK_QUEUE_COMPUTE_BIT, "compute")
        PRINT_FLAG(VK_QUEUE_TRANSFER_BIT, "transfer")
        PRINT_FLAG(VK_QUEUE_SPARSE_BINDING_BIT, "sparse")
#undef PRINT_FLAG

        if (flags != 0) {
            if (flags != properties.queueFlags)
                printf("|");
            printf("0x%x", flags);
        }
        printf("\n");

        printf("   Count: %u\n", properties.queueCount);
    }

    if (computeQueueFamily < 0) {
        fprintf(stderr, "vulkan: did not find a compute queue!\n");
        exit(1);
    }

    printf("vulkan: choosing compute queue family index %u\n", computeQueueFamily);

    if (transferQueueFamily >= 0) {
        printf("vulkan: choosing dedicated transfer queue family index %u\n", transferQueueFamily);
    } else if (queueFamilyProperties[computeQueueFamily].queueCount >= 2) {
        printf("vulkan: choosing transfer queue family index %u\n", computeQueueFamily);
        transferQueueFamily = computeQueueFamily;
    } else {
        printf("vulkan: no separate transfer queue\n");
    }

    const float queuePriorities[] = { 1.0, 1.0 };
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

    {
        queueCreateInfos.emplace_back();
        auto &queueCreateInfo = queueCreateInfos.back();
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = computeQueueFamily;
        queueCreateInfo.queueCount = computeQueueFamily == transferQueueFamily ? 2 : 1;
        queueCreateInfo.pQueuePriorities = queuePriorities;
    }

    if (transferQueueFamily >= 0 && transferQueueFamily != computeQueueFamily) {
        queueCreateInfos.emplace_back();
        auto &queueCreateInfo = queueCreateInfos.back();
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = transferQueueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = queuePriorities;
    }

    VkPhysicalDeviceVulkan13Features vulkan13Features = {};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.computeFullSubgroups = true;

    VkPhysicalDeviceVulkan12Features vulkan12Features = {};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.pNext = &vulkan13Features;
    vulkan12Features.shaderFloat16 = true;

    VkPhysicalDeviceVulkan11Features vulkan11Features = {};
    vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan11Features.pNext = &vulkan12Features;
    vulkan11Features.storageBuffer16BitAccess = true;

    VkPhysicalDeviceFeatures2 features = {};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &vulkan11Features;
    features.features.shaderInt16 = true;

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &features;
    createInfo.queueCreateInfoCount = queueCreateInfos.size();
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    OwnedDevice ownedDevice;
    ownedDevice.vk = &vk;
    LLVK_CHECK_RESULT(vk.CreateDevice(physicalDevice, &createInfo, NULL,
                                      &ownedDevice.device));

    vk.GetDeviceQueue(ownedDevice.device, computeQueueFamily, 0, &ownedDevice.computeQueue);
    if (transferQueueFamily >= 0) {
        unsigned index = computeQueueFamily == transferQueueFamily ? 1 : 0;
        vk.GetDeviceQueue(ownedDevice.device, transferQueueFamily, index,
                          &ownedDevice.transferQueue);
    }

    return std::move(ownedDevice);
}

struct SpecConstants {
    uint nEmbd = 6656;
    uint nCtx = 2048;
    float rotaryTheta = 10000.0;
};
static const VkSpecializationMapEntry g_specMapEntries[] = {
    { 0, offsetof(SpecConstants, nEmbd), sizeof(SpecConstants::nEmbd) },
    { 1, offsetof(SpecConstants, nCtx), sizeof(SpecConstants::nCtx) },
    { 2, offsetof(SpecConstants, rotaryTheta), sizeof(SpecConstants::rotaryTheta) },
};

static const VkDescriptorSetLayoutBinding g_dsetLayoutGlobal[] = {
    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // constants
    { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // history indices
};

static const VkDescriptorSetLayoutBinding g_dsetLayoutPerKernel[] = {
    { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // input
    { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // output
};

static const VkDescriptorSetLayoutBinding g_dsetLayoutPerLayer[] = {
    { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // attention norm
    { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // Wq
    { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // Wk
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // Wv
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // Wo
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // key cache
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // value cache
};

class LlamaContext {
public:
    LlamaContext(Instance &vk, Device &device, const std::string &modelPath, const llama_file_info &fileInfo);
    ~LlamaContext();

private:
    VkDescriptorSetLayout createDescriptorSetLayoutImpl(const VkDescriptorSetLayoutBinding *bindings, size_t count);
    template <int N>
    VkDescriptorSetLayout createDescriptorSetLayout(const VkDescriptorSetLayoutBinding (&bindings)[N]) {
        return createDescriptorSetLayoutImpl(bindings, N);
    }
    VkPipeline createPipeline(const std::string &kernelName);
    void destroy();

    Instance &vk;
    Device &device;

    unsigned m_numLayers;

    VkPipeline m_kernelThinFp16Attention = nullptr;
    VkPipeline m_kernelThinFp16RmsNorm = nullptr;

    VkPipelineLayout m_pipelineLayout = nullptr;

    VkDescriptorSetLayout m_dsetLayoutGlobal = nullptr;
    VkDescriptorSetLayout m_dsetLayoutPerKernel = nullptr;
    VkDescriptorSetLayout m_dsetLayoutPerLayer = nullptr;

    VkDescriptorPool m_descriptorPool = nullptr;

    std::string m_modelPath;
    llama_file_info m_fileInfo;

    SpecConstants m_specData;
    VkSpecializationInfo m_specInfo;
};

LlamaContext::LlamaContext(Instance &vk, Device &device, const std::string &modelPath, const llama_file_info &fileInfo)
    : vk(vk), device(device), m_modelPath(modelPath), m_fileInfo(fileInfo)
{
    m_numLayers = fileInfo.n_layer;
    m_specData.nEmbd = fileInfo.n_embd;

    m_specInfo.mapEntryCount = sizeof(g_specMapEntries) / sizeof(g_specMapEntries[0]);
    m_specInfo.pMapEntries = g_specMapEntries;
    m_specInfo.dataSize = sizeof(m_specData);
    m_specInfo.pData = &m_specData;

    m_dsetLayoutGlobal = createDescriptorSetLayout(g_dsetLayoutGlobal);
    m_dsetLayoutPerKernel = createDescriptorSetLayout(g_dsetLayoutPerKernel);
    m_dsetLayoutPerLayer = createDescriptorSetLayout(g_dsetLayoutPerLayer);

    const VkDescriptorSetLayout setLayouts[] = {
        m_dsetLayoutGlobal,
        m_dsetLayoutPerKernel,
        m_dsetLayoutPerLayer,
    };

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = sizeof(setLayouts) / sizeof(setLayouts[0]);
    pipelineLayoutCreateInfo.pSetLayouts = setLayouts;
    LLVK_CHECK_RESULT(vk.CreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &m_pipelineLayout));

    unsigned globalSets = 2;
    unsigned layerSets = 2 * m_numLayers;
    unsigned kernelSets =
        2 * (
            0 +
            m_numLayers * 2
        );

    const VkDescriptorPoolSize poolSizes[] = {
        {
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                globalSets * 1,
        },
        {
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                (uint32_t)(
                    globalSets * 1
                    + layerSets * sizeof(g_dsetLayoutPerLayer) / sizeof(g_dsetLayoutPerLayer[0])
                    + kernelSets * sizeof(g_dsetLayoutPerKernel) / sizeof(g_dsetLayoutPerKernel[0])
                ),
        },
    };

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = globalSets + layerSets + kernelSets;
    descriptorPoolCreateInfo.poolSizeCount = sizeof(poolSizes) / sizeof(poolSizes[0]);
    descriptorPoolCreateInfo.pPoolSizes = poolSizes;
    LLVK_CHECK_RESULT(vk.CreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &m_descriptorPool));

    m_kernelThinFp16RmsNorm = createPipeline("KernelThinFp16RmsNorm");
    m_kernelThinFp16Attention = createPipeline("KernelThinFp16Attention");
}

LlamaContext::~LlamaContext() {
    destroy();
}

void LlamaContext::destroy() {
    if (m_descriptorPool) {
        vk.DestroyDescriptorPool(device, m_descriptorPool, nullptr);
        m_descriptorPool = nullptr;
    }

    if (m_pipelineLayout) {
        vk.DestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = nullptr;
    }

#define DESTROY(name) \
    if (name) { \
        vk.DestroyDescriptorSetLayout(device, name, nullptr); \
        name = nullptr; \
    }

    DESTROY(m_dsetLayoutGlobal);
    DESTROY(m_dsetLayoutPerKernel);
    DESTROY(m_dsetLayoutPerLayer);

#undef DESTROY
#define DESTROY(name) \
    if (name) { \
        vk.DestroyPipeline(device, name, nullptr); \
        name = nullptr; \
    }

    DESTROY(m_kernelThinFp16Attention);
    DESTROY(m_kernelThinFp16RmsNorm);

#undef DESTROY
}

VkDescriptorSetLayout LlamaContext::createDescriptorSetLayoutImpl(
        const VkDescriptorSetLayoutBinding *bindings, size_t count) {
    VkDescriptorSetLayoutCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    createInfo.bindingCount = count;
    createInfo.pBindings = bindings;

    VkDescriptorSetLayout setLayout;
    LLVK_CHECK_RESULT(vk.CreateDescriptorSetLayout(device, &createInfo, nullptr, &setLayout));

    return setLayout;
}

VkPipeline LlamaContext::createPipeline(const std::string &kernelName) {
    std::string spvName = "vulkan/" + kernelName + ".spv";
    std::ifstream fin(spvName.c_str(), std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, spvName.c_str());
        exit(1);
    }

    std::vector<char> buf;
    fin.seekg(0, std::ios::end);
    buf.resize(fin.tellg());
    fin.seekg(0);
    fin.read(buf.data(), buf.size());

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = buf.size();
    shaderModuleCreateInfo.pCode = (uint32_t *)buf.data();

    VkShaderModule shaderModule;
    LLVK_CHECK_RESULT(vk.CreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule));

    VkComputePipelineCreateInfo computePipelineCreateInfo = {};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computePipelineCreateInfo.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    computePipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computePipelineCreateInfo.stage.module = shaderModule;
    computePipelineCreateInfo.stage.pName = kernelName.c_str();
    computePipelineCreateInfo.stage.pSpecializationInfo = &m_specInfo;
    computePipelineCreateInfo.layout = m_pipelineLayout;

    VkPipeline pipeline;
    LLVK_CHECK_RESULT(vk.CreateComputePipelines(device, nullptr, 1, &computePipelineCreateInfo, nullptr, &pipeline));

    vk.DestroyShaderModule(device, shaderModule, nullptr);

    return pipeline;
}

} // namespace llvk

struct llvk_params {
    std::string model;
    int32_t n_parts = -1;   // amount of model parts (-1 = determine from model dimensions)

    // sampling parameters
    int32_t seed = -1;
    int32_t top_k = 40;
    float   top_p = 0.95f;
    float   temp  = 0.80f;
    int32_t repeat_last_n = 64;   // last n tokens to penalize
    float   repeat_penalty  = 1.10f;

    // driver parameters
    std::string prompt = "";
    int32_t n_predict = 128; // max. num tokens to predict

    bool memory_f16        = true;  // use f16 instead of f32 for memory kv
    bool use_color         = false; // use color to distinguish generations and inputs
    bool interactive       = false; // interactive mode

    bool embedding         = false; // get only sentence embedding
    bool interactive_start = false; // wait for user input immediately

    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool ignore_eos        = false; // do not stop generating after eos
    bool perplexity        = false; // compute perplexity over the prompt
    bool mem_test          = false; // compute maximum memory usage
    bool verbose_prompt    = false; // print prompt tokens before generation
};

static void print_usage(int /*argc*/, char ** argv, const llvk_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -i, --interactive     run in interactive mode\n");
    fprintf(stderr, "  --interactive-first   run in interactive mode and wait for input right away\n");
    fprintf(stderr, "  -ins, --instruct      run in instruction mode (use with Alpaca models)\n");
    fprintf(stderr, "  --color               colorise output to distinguish prompt and user input from generations\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for <= 0)\n");
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: empty)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME\n");
    fprintf(stderr, "                        prompt file to start generation.\n");
    fprintf(stderr, "  -n N, --n_predict N   number of tokens to predict (default: %d, -1 - infinity)\n", params.n_predict);
    fprintf(stderr, "  --top_k N             top-k sampling (default: %d)\n", params.top_k);
    fprintf(stderr, "  --top_p N             top-p sampling (default: %.1f)\n", params.top_p);
    fprintf(stderr, "  --repeat_last_n N     last n tokens to consider for penalize (default: %d)\n", params.repeat_last_n);
    fprintf(stderr, "  --repeat_penalty N    penalize repeat sequence of tokens (default: %.1f)\n", params.repeat_penalty);
    fprintf(stderr, "  --ignore-eos          ignore end of stream token and continue generating\n");
    fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", params.temp);
    fprintf(stderr, "  --n_parts N           number of model parts (default: -1 = determine from dimensions)\n");
    fprintf(stderr, "  --perplexity          compute perplexity over the prompt\n");
    fprintf(stderr, "  --mtest               compute maximum memory usage\n");
    fprintf(stderr, "  --verbose-prompt      print prompt before generation\n");
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path\n");
    fprintf(stderr, "\n");
}

static void params_parse(int argc, char ** argv, llvk_params &params) {
    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.seed = std::stoi(argv[i]);
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.prompt = argv[i];
        } else if (arg == "-f" || arg == "--file") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::ifstream file(argv[i]);
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        } else if (arg == "-n" || arg == "--n_predict") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_predict = std::stoi(argv[i]);
        } else if (arg == "--top_k") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.top_k = std::stoi(argv[i]);
        } else if (arg == "--top_p") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.top_p = std::stof(argv[i]);
        } else if (arg == "--temp") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.temp = std::stof(argv[i]);
        } else if (arg == "--repeat_last_n") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.repeat_last_n = std::stoi(argv[i]);
        } else if (arg == "--repeat_penalty") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.repeat_penalty = std::stof(argv[i]);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "-i" || arg == "--interactive") {
            params.interactive = true;
        } else if (arg == "--embedding") {
            params.embedding = true;
        } else if (arg == "--interactive-start") {
            params.interactive = true;
        } else if (arg == "--interactive-first") {
            params.interactive_start = true;
        } else if (arg == "-ins" || arg == "--instruct") {
            params.instruct = true;
        } else if (arg == "--color") {
            params.use_color = true;
        } else if (arg == "--mtest") {
            params.mem_test = true;
        } else if (arg == "--verbose-prompt") {
            params.verbose_prompt = true;
        } else if (arg == "--perplexity") {
            params.perplexity = true;
        } else if (arg == "--ignore-eos") {
            params.ignore_eos = true;
        } else if (arg == "--n_parts") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_parts = std::stoi(argv[i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(1);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv, params);
        exit(1);
    }

    if (params.model.empty()) {
        fprintf(stderr, "Specify the model using '-m' or '--model'.\n");
        exit(1);
    }
}

static std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    // initialize to prompt number of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text.size() + (int)add_bos);
    int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);
    return res;
}

int main(int argc, char **argv) {
    llvk_params params;
    params_parse(argc, argv, params);

    auto vk = llvk::OwnedInstance::createDefault();

    auto device = llvk::OwnedDevice::createDefault(vk);

    llama_context_params ctx_params = {};
    ctx_params.n_ctx = 2048;
    ctx_params.n_parts = params.n_parts;
    ctx_params.seed = params.seed;
    ctx_params.vocab_only = true;

    llama_file_info model_file_info;
    llama_context *ctx = llama_init_from_file(params.model.c_str(), ctx_params, &model_file_info);
    if (!ctx)
        exit(1);

    llvk::LlamaContext vkctx(vk, device, params.model, model_file_info);

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // Tokenize the prompt
    auto embd_inp = llama_tokenize(ctx, params.prompt.c_str(), true);

    printf("Initial embd_inp:\n");
    for (const auto &token : embd_inp)
        printf("  %u: '%s'\n", token, llama_token_to_str(ctx, token));
    printf("--\n");

    printf("Hi rebuild\n");
    return 0;
}
