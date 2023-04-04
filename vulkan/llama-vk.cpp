
#include <array>
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../llama.h"

#include "vulkan/vulkan.h"

#define LLAMA_HOST
#include "llama-vk-shader.h"

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

template <typename T, typename U>
static T alignPot(T x, U to) {
    return (x + T(to - 1)) & ~T(to - 1);
}

static std::string formatNumBytes(uint64_t numBytes) {
    static const char *units[] = {
        "B", "kiB", "MiB", "GiB", "TiB", nullptr,
    };
    uint64_t quantity = numBytes;
    unsigned unit = 0;

    while (units[unit + 1] && quantity >= 4 * 1024) {
        quantity = (quantity + 1024 - 1) / 1024;
        ++unit;
    }

    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f%s", (float)quantity, units[unit]);
    return buf;
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

class Device;

class MemoryAndBuffer {
    friend class Device;
private:
    Device *m_device = nullptr;
    VkBuffer m_buffer = nullptr;
    VkDeviceMemory m_memory = nullptr;

public:
    MemoryAndBuffer() {}
    MemoryAndBuffer(Device *device, VkBuffer buffer, VkDeviceMemory memory)
        : m_device(device), m_buffer(buffer), m_memory(memory) {}
    ~MemoryAndBuffer();

    MemoryAndBuffer(MemoryAndBuffer &&rhs);
    MemoryAndBuffer &operator=(MemoryAndBuffer &&rhs);

    void reset();

    bool valid() const { return m_buffer; }

    VkBuffer buffer() { return m_buffer; }
    VkDeviceMemory memory() { return m_memory; }

private:
    MemoryAndBuffer(const MemoryAndBuffer &rhs) = delete;
    MemoryAndBuffer &operator=(const MemoryAndBuffer &rhs) = delete;
};

class Device {
protected:
    Instance *vk = nullptr;
    VkDevice device = nullptr;
    VkQueue computeQueue = nullptr;
    VkQueue transferQueue = nullptr;
    VkPhysicalDeviceMemoryProperties m_memoryProperties;
    int m_deviceMemType = -1;
    int m_hostMemType = -1;

    Device() = default;

public:
    Device(Instance &vk, VkDevice device, VkQueue computeQueue, VkQueue transferQueue)
        : vk(&vk), device(device), computeQueue(computeQueue)
        , transferQueue(transferQueue) {}

    void init(VkPhysicalDevice physicalDevice);

    operator VkDevice() { return device; }
    Instance &instance() { return *vk; }

    MemoryAndBuffer allocateDevice(uint64_t numBytes) { return allocate(m_deviceMemType, numBytes); }
    MemoryAndBuffer allocateHost(uint64_t numBytes) { return allocate(m_hostMemType, numBytes); }

    void allocateDescriptorSets(VkDescriptorPool pool, VkDescriptorSetLayout layout,
                                unsigned count, VkDescriptorSet *sets);

private:
    MemoryAndBuffer allocate(int memType, uint64_t numBytes);
};

void Device::init(VkPhysicalDevice physicalDevice) {
    vk->GetPhysicalDeviceMemoryProperties(physicalDevice, &m_memoryProperties);

    int bestDeviceMemType = -1;
    uint64_t bestDeviceHeapSize = 0;
    int bestHostMemType = -1;
    uint64_t bestHostHeapSize = 0;
    bool bestHostIsLocal = false;

    for (unsigned i = 0; i < m_memoryProperties.memoryTypeCount; ++i) {
        const VkMemoryType &type = m_memoryProperties.memoryTypes[i];
        const VkMemoryHeap &heap = m_memoryProperties.memoryHeaps[type.heapIndex];

        if (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            if (bestDeviceMemType < 0 || heap.size > bestDeviceHeapSize) {
                bestDeviceMemType = i;
                bestDeviceHeapSize = heap.size;
            }
        }

        if ((type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
            (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT))
        {
            bool isLocal = !(type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (bestHostMemType < 0 ||
                (isLocal && !bestHostIsLocal) ||
                (isLocal == bestHostIsLocal && heap.size > bestHostHeapSize))
            {
                bestHostMemType = i;
                bestHostHeapSize = heap.size;
                bestHostIsLocal = isLocal;
            }
        }
    }

    if (bestDeviceMemType < 0) {
        fprintf(stderr, "%s: no suitable GPU memory found\n", __func__);
        exit(1);
    }

    if (bestHostMemType < 0) {
        fprintf(stderr, "%s: no suitable CPU memory found\n", __func__);
        exit(1);
    }

    printf("vulkan: using GPU memory type %u and CPU memory type %u\n",
            bestDeviceMemType, bestHostMemType);

    m_deviceMemType = bestDeviceMemType;
    m_hostMemType = bestHostMemType;
}

MemoryAndBuffer Device::allocate(int memType, uint64_t numBytes) {
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = numBytes;
    bufferCreateInfo.usage =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VkBuffer buffer;
    LLVK_CHECK_RESULT(vk->CreateBuffer(device, &bufferCreateInfo, nullptr, &buffer));

    VkMemoryRequirements requirements;
    vk->GetBufferMemoryRequirements(device, buffer, &requirements);

    if ((requirements.memoryTypeBits & (1 << memType)) == 0) {
        fprintf(stderr, "%s: can't allocate buffer in the desired memory type\n", __func__);
        return {};
    }

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.memoryTypeIndex = memType;
    allocateInfo.allocationSize = requirements.size;

    VkDeviceMemory memory;
    VkResult result = vk->AllocateMemory(device, &allocateInfo, nullptr, &memory);
    if (result != VK_SUCCESS) {
        const char *errorStr = "unknown";
        switch (result) {
        case VK_ERROR_OUT_OF_HOST_MEMORY: errorStr = "out of host (CPU) memory"; break;
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: errorStr = "out of device (GPU) memory"; break;
        default: break;
        }

        fprintf(stderr, "%s: attempting to allocate %s failed.\n", __func__, formatNumBytes(numBytes).c_str());
        fprintf(stderr, "%s: error: %s (%d)\n", __func__, errorStr, result);

        vk->DestroyBuffer(device, buffer, nullptr);
        return {};
    }

    LLVK_CHECK_RESULT(vk->BindBufferMemory(device, buffer, memory, 0));

    return {this, buffer, memory};
}

void Device::allocateDescriptorSets(VkDescriptorPool pool, VkDescriptorSetLayout layout,
                                    unsigned count, VkDescriptorSet *sets) {
    std::vector<VkDescriptorSetLayout> layouts(count, layout);

    VkDescriptorSetAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = pool;
    allocateInfo.descriptorSetCount = count;
    allocateInfo.pSetLayouts = layouts.data();
    LLVK_CHECK_RESULT(vk->AllocateDescriptorSets(device, &allocateInfo, sets));
}

MemoryAndBuffer::~MemoryAndBuffer() {
    reset();
}

MemoryAndBuffer::MemoryAndBuffer(MemoryAndBuffer &&rhs) : MemoryAndBuffer() {
    *this = std::move(rhs);
}

MemoryAndBuffer &MemoryAndBuffer::operator=(MemoryAndBuffer &&rhs) {
    if (this != &rhs) {
        reset();

        m_device = rhs.m_device;
        m_buffer = rhs.m_buffer;
        m_memory = rhs.m_memory;

        rhs.m_buffer = nullptr;
        rhs.m_memory = nullptr;
    }
    return *this;
}

void MemoryAndBuffer::reset() {
    if (m_buffer) {
        auto &vk = m_device->instance();
        vk.FreeMemory(*m_device, m_memory, nullptr);
        vk.DestroyBuffer(*m_device, m_buffer, nullptr);

        m_buffer = nullptr;
        m_memory = nullptr;
    }
}

struct Range {
    uint64_t offset = 0;
    uint64_t range = 0;
};

class WriteBufferDescriptors {
public:
    WriteBufferDescriptors(Device &device) : m_device(device) {}
    ~WriteBufferDescriptors() {
        // Must explicitly commit
        assert(m_descriptorWrites.empty());
    }

    void commit();
    void reset();

    void write(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
               VkBuffer buffer, Range range);
    void writeUniform(VkDescriptorSet set, uint32_t binding, VkBuffer buffer, Range range);
    void writeStorage(VkDescriptorSet set, uint32_t binding, VkBuffer buffer, Range range);

private:
    Device &m_device;
    std::vector<VkDescriptorBufferInfo> m_bufferInfos;
    std::vector<VkWriteDescriptorSet> m_descriptorWrites;
    std::vector<size_t> m_offsets;
};

void WriteBufferDescriptors::commit() {
    if (m_descriptorWrites.empty())
        return;

    for (size_t i = 0; i < m_descriptorWrites.size(); ++i) {
        m_descriptorWrites[i].pBufferInfo = &m_bufferInfos[m_offsets[i]];
    }

    m_device.instance().UpdateDescriptorSets(
        m_device, m_descriptorWrites.size(), m_descriptorWrites.data(), 0, nullptr);

    reset();
}

void WriteBufferDescriptors::reset() {
    m_bufferInfos.clear();
    m_descriptorWrites.clear();
    m_offsets.clear();
}

void WriteBufferDescriptors::write(
        VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
        VkBuffer buffer, Range range) {
    VkDescriptorBufferInfo bufferInfo;
    bufferInfo.buffer = buffer;
    bufferInfo.offset = range.offset;
    bufferInfo.range = range.range;

    if (!m_descriptorWrites.empty()) {
        auto &write = m_descriptorWrites.back();
        if (write.descriptorType == type && write.dstSet == set &&
            write.dstBinding + write.descriptorCount == binding)
        {
            m_bufferInfos.push_back(bufferInfo);
            write.descriptorCount++;
            return;
        }
    }

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorType = type;
    write.descriptorCount = 1;
    m_descriptorWrites.push_back(write);

    m_offsets.push_back(m_bufferInfos.size());
    m_bufferInfos.push_back(bufferInfo);
}

void WriteBufferDescriptors::writeUniform(
        VkDescriptorSet set, uint32_t binding,
        VkBuffer buffer, Range range) {
    write(set, binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, buffer, range);
}

void WriteBufferDescriptors::writeStorage(
        VkDescriptorSet set, uint32_t binding,
        VkBuffer buffer, Range range) {
    write(set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer, range);
}

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

    ownedDevice.init(physicalDevice);

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
    LlamaContext(Instance &vk, Device &device, const std::string &modelPath, const llama_file_info &fileInfo,
                 unsigned maxCacheEntries);
    ~LlamaContext();

private:
    VkDescriptorSetLayout createDescriptorSetLayoutImpl(const VkDescriptorSetLayoutBinding *bindings, size_t count);
    template <int N>
    VkDescriptorSetLayout createDescriptorSetLayout(const VkDescriptorSetLayoutBinding (&bindings)[N]) {
        return createDescriptorSetLayoutImpl(bindings, N);
    }
    VkPipeline createPipeline(const std::string &kernelName);
    void destroy();

    uint64_t calcQ4Size(unsigned quantizedDim, unsigned other);

    Instance &vk;
    Device &device;

    unsigned m_numVocab;
    unsigned m_numLayers;
    unsigned m_maxCacheEntries;

    VkPipeline m_kernelThinFp16Attention = nullptr;
    VkPipeline m_kernelThinFp16RmsNorm = nullptr;

    VkPipelineLayout m_pipelineLayout = nullptr;

    VkDescriptorSetLayout m_dsetLayoutGlobal = nullptr;
    VkDescriptorSetLayout m_dsetLayoutPerKernel = nullptr;
    VkDescriptorSetLayout m_dsetLayoutPerLayer = nullptr;

    VkDescriptorPool m_descriptorPool = nullptr;

    VkDescriptorSet m_dsetGlobal[2] = {};
    VkDescriptorSet m_dsetKernel[3] = {};
    std::vector<VkDescriptorSet> m_dsetLayer;

    struct {
        uint64_t size = 0;
        Range constants[2];
        Range historyIndex;
        Range embedding;
        Range activations[3];
    } m_globalOffsets;
    MemoryAndBuffer m_globalMemory;

    struct {
        uint64_t size = 0;
        Range attentionNorm;
        Range Wq;
        Range Wk;
        Range Wv;
        Range Wo;
        Range cacheKeys;
        Range cacheValues;
    } m_layerOffsets;
    std::vector<MemoryAndBuffer> m_layerMemory;

    VkDeviceMemory m_hostMemory;

    std::string m_modelPath;
    llama_file_info m_fileInfo;

    SpecConstants m_specData;
    VkSpecializationInfo m_specInfo;
};

LlamaContext::LlamaContext(Instance &vk, Device &device, const std::string &modelPath, const llama_file_info &fileInfo,
                           unsigned maxCacheEntries)
    : vk(vk), device(device), m_modelPath(modelPath), m_fileInfo(fileInfo)
{
    m_numVocab = fileInfo.n_vocab;
    m_numLayers = fileInfo.n_layer;
    m_maxCacheEntries = maxCacheEntries;
    m_specData.nEmbd = fileInfo.n_embd;

    // Step 1: Descriptor set and pipeline layouts and pipelines
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

    m_kernelThinFp16RmsNorm = createPipeline("KernelThinFp16RmsNorm");
    m_kernelThinFp16Attention = createPipeline("KernelThinFp16Attention");

    // Step 2: Memory allocation
    //
    // Allocate memory in per-layer chunks since some implementations can't
    // allocate more than ~2 GiB in a single allocation.
    {
#define OFFSET(memory, sub, bytes) \
        do { \
            memory.sub.offset = memory.size; \
            memory.sub.range = (bytes); \
            memory.size += alignPot((bytes), 256); \
        } while (false)

        OFFSET(m_layerOffsets, attentionNorm, 2 * m_specData.nEmbd);

        uint64_t attnMatrixSize = calcQ4Size(m_specData.nEmbd, m_specData.nEmbd);
        OFFSET(m_layerOffsets, Wq, attnMatrixSize);
        OFFSET(m_layerOffsets, Wk, attnMatrixSize);
        OFFSET(m_layerOffsets, Wv, attnMatrixSize);
        OFFSET(m_layerOffsets, Wo, attnMatrixSize);

        uint64_t cacheSize = 2 * m_specData.nEmbd * m_maxCacheEntries;
        OFFSET(m_layerOffsets, cacheKeys, cacheSize);
        OFFSET(m_layerOffsets, cacheValues, cacheSize);

        OFFSET(m_globalOffsets, constants[0], sizeof(shader::GlobalConstantBuffer));
        OFFSET(m_globalOffsets, constants[1], sizeof(shader::GlobalConstantBuffer));
        OFFSET(m_globalOffsets, historyIndex, 2 * 2048);

        uint64_t embdMatrixSize = calcQ4Size(m_specData.nEmbd, m_numVocab);
        OFFSET(m_globalOffsets, embedding, embdMatrixSize);

        uint64_t activationSize = 2 * m_specData.nEmbd;
        OFFSET(m_globalOffsets, activations[0], activationSize);
        OFFSET(m_globalOffsets, activations[1], activationSize);
        OFFSET(m_globalOffsets, activations[2], activationSize);

        printf("vulkan: allocating %s of device memory\n",
               formatNumBytes(m_globalOffsets.size + m_numLayers * m_layerOffsets.size).c_str());

        m_globalMemory = device.allocateDevice(m_globalOffsets.size);
        if (!m_globalMemory.valid())
            exit(1);

        for (unsigned i = 0; i < m_numLayers; ++i) {
            auto memory = device.allocateDevice(m_layerOffsets.size);
            if (!memory.valid())
                exit(1);
            m_layerMemory.push_back(std::move(memory));
        }
    }

    // Step 3: Set up descriptor sets
    {
        unsigned globalSets = sizeof(m_dsetGlobal) / sizeof(m_dsetGlobal[0]); // Double buffer
        unsigned layerSets = m_numLayers;
        unsigned kernelSets = sizeof(m_dsetKernel) / sizeof(m_dsetKernel[0]); // Triple buffer(?)

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

        device.allocateDescriptorSets(m_descriptorPool, m_dsetLayoutGlobal, globalSets, m_dsetGlobal);
        device.allocateDescriptorSets(m_descriptorPool, m_dsetLayoutPerKernel, kernelSets, m_dsetKernel);
        m_dsetLayer.resize(m_numLayers);
        device.allocateDescriptorSets(m_descriptorPool, m_dsetLayoutPerLayer, m_numLayers, m_dsetLayer.data());

        WriteBufferDescriptors write(device);
        for (unsigned i = 0; i < globalSets; ++i) {
            write.writeUniform(m_dsetGlobal[i], 0, m_globalMemory.buffer(), m_globalOffsets.constants[i]);
            write.writeStorage(m_dsetGlobal[i], 1, m_globalMemory.buffer(), m_globalOffsets.historyIndex);
        }

        for (unsigned i = 0; i < kernelSets; ++i) {
            write.writeStorage(m_dsetKernel[i], 0, m_globalMemory.buffer(), m_globalOffsets.activations[i]);
            write.writeStorage(m_dsetKernel[i], 1, m_globalMemory.buffer(), m_globalOffsets.activations[(i + 1) % kernelSets]);
        }

        for (unsigned i = 0; i < m_numLayers; ++i) {
            write.writeStorage(m_dsetLayer[i], 0, m_layerMemory[i].buffer(), m_layerOffsets.attentionNorm);
            write.writeStorage(m_dsetLayer[i], 1, m_layerMemory[i].buffer(), m_layerOffsets.Wq);
            write.writeStorage(m_dsetLayer[i], 2, m_layerMemory[i].buffer(), m_layerOffsets.Wk);
            write.writeStorage(m_dsetLayer[i], 3, m_layerMemory[i].buffer(), m_layerOffsets.Wv);
            write.writeStorage(m_dsetLayer[i], 4, m_layerMemory[i].buffer(), m_layerOffsets.Wo);
            write.writeStorage(m_dsetLayer[i], 5, m_layerMemory[i].buffer(), m_layerOffsets.cacheKeys);
            write.writeStorage(m_dsetLayer[i], 6, m_layerMemory[i].buffer(), m_layerOffsets.cacheValues);
        }

        write.commit();
    }
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
    do { \
        if (name) { \
            vk.DestroyDescriptorSetLayout(device, name, nullptr); \
            name = nullptr; \
        } \
    } while(false)

    DESTROY(m_dsetLayoutGlobal);
    DESTROY(m_dsetLayoutPerKernel);
    DESTROY(m_dsetLayoutPerLayer);

#undef DESTROY
#define DESTROY(name) \
    do { \
        if (name) { \
            vk.DestroyPipeline(device, name, nullptr); \
            name = nullptr; \
        } \
    } while(false)

    DESTROY(m_kernelThinFp16Attention);
    DESTROY(m_kernelThinFp16RmsNorm);

#undef DESTROY
}

uint64_t LlamaContext::calcQ4Size(unsigned quantizedDim, unsigned other) {
    unsigned numBlocks = (quantizedDim + 63) / 64;
    unsigned blockSize = 4 + 32;
    return blockSize * numBlocks * other;
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

    llvk::LlamaContext vkctx(vk, device, params.model, model_file_info, 2048);

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // Tokenize the prompt
    auto embd_inp = llama_tokenize(ctx, params.prompt.c_str(), true);

    printf("Initial embd_inp:\n");
    for (const auto &token : embd_inp)
        printf("  %u: '%s'\n", token, llama_token_to_str(ctx, token));
    printf("--\n");

    return 0;
}
