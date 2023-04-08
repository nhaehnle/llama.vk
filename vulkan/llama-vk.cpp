
#include <array>
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
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

    void utilCmdPipelineMemoryBarrier(
            VkCommandBuffer cmdBuf,
            VkPipelineStageFlags2 srcStageMask, VkAccessFlags2 srcAccessMask,
            VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 dstAccessMask);

#define FN(name) PFN_vk ## name name = nullptr;
#include "llama-vk-functions.inc"
};

void Instance::utilCmdPipelineMemoryBarrier(
        VkCommandBuffer cmdBuf,
        VkPipelineStageFlags2 srcStageMask, VkAccessFlags2 srcAccessMask,
        VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 dstAccessMask) {
    VkMemoryBarrier2 memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    memoryBarrier.srcStageMask = srcStageMask;
    memoryBarrier.srcAccessMask = srcAccessMask;
    memoryBarrier.dstStageMask = dstStageMask;
    memoryBarrier.dstAccessMask = dstAccessMask;

    VkDependencyInfo dependencyInfo = {};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.memoryBarrierCount = 1;
    dependencyInfo.pMemoryBarriers = &memoryBarrier;

    CmdPipelineBarrier2(cmdBuf, &dependencyInfo);
}

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

    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "llama-vk";
    applicationInfo.applicationVersion = 0;
    applicationInfo.apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &applicationInfo;

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

    void *map(uint64_t offset, uint64_t size);
    void unmap();

private:
    MemoryAndBuffer(const MemoryAndBuffer &rhs) = delete;
    MemoryAndBuffer &operator=(const MemoryAndBuffer &rhs) = delete;
};

class TimelineSemaphore {
private:
    Device *m_device = nullptr;
    VkSemaphore m_semaphore = nullptr;

public:
    TimelineSemaphore() {}
    TimelineSemaphore(Device *device, VkSemaphore semaphore)
        : m_device(device), m_semaphore(semaphore) {}
    ~TimelineSemaphore();

    TimelineSemaphore(TimelineSemaphore &&rhs);
    TimelineSemaphore &operator=(TimelineSemaphore &&rhs);

    void reset();

    bool valid() const { return m_semaphore; }
    VkSemaphore semaphore() { return m_semaphore; }

    bool wait(uint64_t value, uint64_t timeoutNs);
    void waitForever(uint64_t value);

private:
    TimelineSemaphore(const TimelineSemaphore &rhs) = delete;
    TimelineSemaphore &operator=(const TimelineSemaphore &rhs) = delete;
};

class Device {
protected:
    Instance *vk = nullptr;
    VkDevice device = nullptr;
    VkQueue m_computeQueue = nullptr;
    VkQueue m_transferQueue = nullptr;
    uint32_t m_computeQueueFamily = ~0;
    uint32_t m_transferQueueFamily = ~0;
    VkPhysicalDeviceMemoryProperties m_memoryProperties;
    int m_deviceMemType = -1;
    int m_hostMemType = -1;

    Device() = default;

public:
    Device(Instance &vk, VkDevice device,
           VkQueue computeQueue, uint32_t computeQueueFamily,
           VkQueue transferQueue, uint32_t transferQueueFamily)
        : vk(&vk), device(device)
        , m_computeQueue(computeQueue), m_computeQueueFamily(computeQueueFamily)
        , m_transferQueue(transferQueue), m_transferQueueFamily(transferQueueFamily)
    {
        assert(computeQueue != nullptr);
    }

    void init(VkPhysicalDevice physicalDevice);

    operator VkDevice() { return device; }
    Instance &instance() { return *vk; }

    bool haveTransferQueue() const { return m_transferQueue != nullptr; }
    VkQueue transferQueue() { assert(haveTransferQueue()); return m_transferQueue; }
    VkQueue computeQueue() { return m_computeQueue; }
    uint32_t transferQueueFamily() const { return m_transferQueueFamily; }
    uint32_t computeQueueFamily() const { return m_computeQueueFamily; }

    MemoryAndBuffer allocateDevice(uint64_t numBytes) { return allocate(m_deviceMemType, numBytes); }
    MemoryAndBuffer allocateHost(uint64_t numBytes) { return allocate(m_hostMemType, numBytes); }

    void allocateDescriptorSets(VkDescriptorPool pool, VkDescriptorSetLayout layout,
                                unsigned count, VkDescriptorSet *sets);

    VkCommandPool createCommandPool(bool transfer);
    TimelineSemaphore createTimelineSemaphore();

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

VkCommandPool Device::createCommandPool(bool transfer) {
    assert(!transfer || haveTransferQueue());

    VkCommandPoolCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    createInfo.queueFamilyIndex = transfer ? m_transferQueueFamily : m_computeQueueFamily;

    VkCommandPool pool;
    LLVK_CHECK_RESULT(vk->CreateCommandPool(device, &createInfo, nullptr, &pool));
    return pool;
}

TimelineSemaphore Device::createTimelineSemaphore() {
    VkSemaphoreTypeCreateInfo typeCreateInfo = {};
    typeCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    typeCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    typeCreateInfo.initialValue = 0;

    VkSemaphoreCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    createInfo.pNext = &typeCreateInfo;

    VkSemaphore semaphore;
    LLVK_CHECK_RESULT(vk->CreateSemaphore(device, &createInfo, nullptr, &semaphore));
    return {this, semaphore};
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
        vk.DestroyBuffer(*m_device, m_buffer, nullptr);
        vk.FreeMemory(*m_device, m_memory, nullptr);

        m_buffer = nullptr;
        m_memory = nullptr;
    }
}

void *MemoryAndBuffer::map(uint64_t offset, uint64_t size) {
    auto &vk = m_device->instance();
    void *ptr;
    LLVK_CHECK_RESULT(vk.MapMemory(*m_device, m_memory, offset, size, 0, &ptr));
    return ptr;
}

void MemoryAndBuffer::unmap() {
    auto &vk = m_device->instance();
    vk.UnmapMemory(*m_device, m_memory);
}

TimelineSemaphore::~TimelineSemaphore() {
    reset();
}

TimelineSemaphore::TimelineSemaphore(TimelineSemaphore &&rhs) : TimelineSemaphore() {
    *this = std::move(rhs);
}

TimelineSemaphore &TimelineSemaphore::operator=(TimelineSemaphore &&rhs) {
    if (this != &rhs) {
        reset();

        m_device = rhs.m_device;
        m_semaphore = rhs.m_semaphore;

        rhs.m_semaphore = nullptr;
    }
    return *this;
}

void TimelineSemaphore::reset() {
    if (m_semaphore) {
        auto &vk = m_device->instance();
        vk.DestroySemaphore(*m_device, m_semaphore, nullptr);
        m_semaphore = nullptr;
    }
}

// Returns true on success, false on timeout.
bool TimelineSemaphore::wait(uint64_t value, uint64_t timeoutNs) {
    auto &vk = m_device->instance();

    VkSemaphoreWaitInfo waitInfo = {};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1,
    waitInfo.pSemaphores = &m_semaphore;
    waitInfo.pValues = &value;

    VkResult result = LLVK_CHECK_RESULT(vk.WaitSemaphores(*m_device, &waitInfo, timeoutNs));
    return result == VK_SUCCESS;
}

void TimelineSemaphore::waitForever(uint64_t value) {
    bool success = wait(value, ~(uint64_t)0);
    if (!success) {
        fprintf(stderr, "%s: timeout\n", __func__);
        exit(1);
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
        FEATURE_CHECK(vulkan12Features, timelineSemaphore)
        FEATURE_CHECK(vulkan13Features, computeFullSubgroups)
        FEATURE_CHECK(vulkan13Features, synchronization2)

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
    vulkan13Features.synchronization2 = true;

    VkPhysicalDeviceVulkan12Features vulkan12Features = {};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.pNext = &vulkan13Features;
    vulkan12Features.shaderFloat16 = true;
    vulkan12Features.timelineSemaphore = true;

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

    vk.GetDeviceQueue(ownedDevice.device, computeQueueFamily, 0, &ownedDevice.m_computeQueue);
    ownedDevice.m_computeQueueFamily = computeQueueFamily;
    if (transferQueueFamily >= 0) {
        unsigned index = computeQueueFamily == transferQueueFamily ? 1 : 0;
        vk.GetDeviceQueue(ownedDevice.device, transferQueueFamily, index,
                          &ownedDevice.m_transferQueue);
        ownedDevice.m_transferQueueFamily = transferQueueFamily;
    }

    ownedDevice.init(physicalDevice);

    return std::move(ownedDevice);
}

// Tensor types on the GPU
enum {
    VKTYPE_F16 = 0,
    VKTYPE_Q4_0_SWZ = 1,
    VKTYPE_Q4_0_LINEAR = 2,
};

uint64_t vktypeSize(int32_t vktype, uint64_t ne0, uint64_t ne1) {
    switch (vktype) {
    case VKTYPE_F16:
        return 2 * ne0 * ne1;
    case VKTYPE_Q4_0_SWZ:
    case VKTYPE_Q4_0_LINEAR:
        assert((ne0 % 64) == 0);
        return (4 + 32) * (ne0 / 64) * ne1;
    }
    abort();
}

struct SpecConstants {
    uint nEmbd = 6656;
    uint nCtx = 2048;
    uint nFF = 17920;
    uint nVocab = 32000;
    uint nHead = 52;
    float rotaryTheta = 10000.0;

    uint mode = 0;
};
static const VkSpecializationMapEntry g_specMapEntries[] = {
    { 0, offsetof(SpecConstants, nEmbd), sizeof(SpecConstants::nEmbd) },
    { 1, offsetof(SpecConstants, nCtx), sizeof(SpecConstants::nCtx) },
    { 2, offsetof(SpecConstants, nFF), sizeof(SpecConstants::nFF) },
    { 3, offsetof(SpecConstants, nVocab), sizeof(SpecConstants::nVocab) },
    { 4, offsetof(SpecConstants, nHead), sizeof(SpecConstants::nHead) },
    { 5, offsetof(SpecConstants, rotaryTheta), sizeof(SpecConstants::rotaryTheta) },
    { 10, offsetof(SpecConstants, mode), sizeof(SpecConstants::mode) },
};

static const VkDescriptorSetLayoutBinding g_dsetLayoutGlobal[] = {
    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // constants
    { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // history indices
    { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // embedding weights
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // activations bypass
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // activations stage1
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // activations stage2
};

static const VkDescriptorSetLayoutBinding g_dsetLayoutPerLayer[] = {
    { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // attention norm
    { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // Wq
    { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // Wk
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // Wv
    { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // Wo
    { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // key cache
    { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // value cache
    { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // FFN norm
    { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // W1
    { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // W2
    { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // W3
};

class LlamaContext {
    struct ModelUploader;
public:
    LlamaContext(Instance &vk, Device &device, const std::string &modelPath, const llama_file_info &fileInfo,
                 unsigned maxCacheEntries);
    ~LlamaContext();

    void uploadModel();

    void process(const llama_token *token, size_t numTokens);

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

    unsigned m_numVocab;
    unsigned m_numLayers;
    unsigned m_maxCacheEntries;

    VkPipeline m_kernelThinFp16Attention = nullptr;
    VkPipeline m_kernelThinFp16FirstRmsNorm = nullptr;
    VkPipeline m_kernelThinFp16MatMulAdd = nullptr;
    VkPipeline m_kernelThinFp16RmsNormAttention = nullptr;
    VkPipeline m_kernelThinFp16RmsNormFFN = nullptr;

    VkPipelineLayout m_pipelineLayout = nullptr;

    VkDescriptorSetLayout m_dsetLayoutGlobal = nullptr;
    VkDescriptorSetLayout m_dsetLayoutPerKernel = nullptr;
    VkDescriptorSetLayout m_dsetLayoutPerLayer = nullptr;

    VkDescriptorPool m_descriptorPool = nullptr;

    VkDescriptorSet m_dsetGlobal[2] = {};
    std::vector<VkDescriptorSet> m_dsetLayer;

    struct {
        uint64_t size = 0;
        Range constants[2];
        Range historyIndex;
        Range embedding;
        Range bypass;
        Range stage1;
        Range stage2;
    } m_globalOffsets;
    MemoryAndBuffer m_globalMemory;

    struct {
        uint64_t size = 0;
        Range attentionNorm;
        Range Wq;
        Range Wk;
        Range Wv;
        Range Wo;
        Range ffnNorm;
        Range W1;
        Range W2;
        Range W3;
        Range cacheKeys;
        Range cacheValues;
    } m_layerOffsets;
    std::vector<MemoryAndBuffer> m_layerMemory;

    struct {
        uint64_t size = 0;
        Range debugActivations;
    } m_hostOffsets;
    MemoryAndBuffer m_hostMemory;

    std::string m_modelPath;
    llama_file_info m_fileInfo;

    SpecConstants m_specData;
    VkSpecializationInfo m_specInfo;

    VkCommandPool m_computeCommandPool = nullptr;
    VkCommandPool m_transferCommandPool = nullptr;

    TimelineSemaphore m_semaphore;
    uint64_t m_numSubmits = 0;
    unsigned m_numPasses = 0;

    VkCommandBuffer m_commandBuffers[2] = {};

    std::unique_ptr<ModelUploader> m_uploader;
};

LlamaContext::LlamaContext(Instance &vk, Device &device, const std::string &modelPath, const llama_file_info &fileInfo,
                           unsigned maxCacheEntries)
    : vk(vk), device(device), m_modelPath(modelPath), m_fileInfo(fileInfo)
{
    m_numVocab = fileInfo.n_vocab;
    m_numLayers = fileInfo.n_layer;
    m_maxCacheEntries = maxCacheEntries;
    m_specData.nEmbd = fileInfo.n_embd;
    m_specData.nFF = fileInfo.n_ff;
    m_specData.nHead = fileInfo.n_head;

    // Step 1: Descriptor set and pipeline layouts and pipelines
    m_specInfo.mapEntryCount = sizeof(g_specMapEntries) / sizeof(g_specMapEntries[0]);
    m_specInfo.pMapEntries = g_specMapEntries;
    m_specInfo.dataSize = sizeof(m_specData);
    m_specInfo.pData = &m_specData;

    m_dsetLayoutGlobal = createDescriptorSetLayout(g_dsetLayoutGlobal);
    m_dsetLayoutPerLayer = createDescriptorSetLayout(g_dsetLayoutPerLayer);

    const VkDescriptorSetLayout setLayouts[] = {
        m_dsetLayoutGlobal,
        m_dsetLayoutPerLayer,
    };

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = sizeof(setLayouts) / sizeof(setLayouts[0]);
    pipelineLayoutCreateInfo.pSetLayouts = setLayouts;
    LLVK_CHECK_RESULT(vk.CreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &m_pipelineLayout));

    m_kernelThinFp16Attention = createPipeline("KernelThinFp16Attention");
    m_specData.mode = 0;
    m_kernelThinFp16FirstRmsNorm = createPipeline("KernelThinFp16FirstRmsNorm");
    m_kernelThinFp16MatMulAdd = createPipeline("KernelThinFp16MatMulAdd");
    m_specData.mode = 0;
    m_kernelThinFp16RmsNormAttention = createPipeline("KernelThinFp16RmsNorm");
    m_specData.mode = 1;
    m_kernelThinFp16RmsNormFFN = createPipeline("KernelThinFp16RmsNorm");

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

        uint64_t attnMatrixSize = vktypeSize(VKTYPE_Q4_0_SWZ, m_specData.nEmbd, m_specData.nEmbd);
        OFFSET(m_layerOffsets, Wq, attnMatrixSize);
        OFFSET(m_layerOffsets, Wk, attnMatrixSize);
        OFFSET(m_layerOffsets, Wv, attnMatrixSize);
        OFFSET(m_layerOffsets, Wo, attnMatrixSize);

        OFFSET(m_layerOffsets, ffnNorm, 2 * m_specData.nEmbd);

        OFFSET(m_layerOffsets, W1, vktypeSize(VKTYPE_Q4_0_SWZ, m_specData.nEmbd, m_specData.nFF));
        OFFSET(m_layerOffsets, W2, vktypeSize(VKTYPE_Q4_0_SWZ, m_specData.nFF, m_specData.nEmbd));
        OFFSET(m_layerOffsets, W3, vktypeSize(VKTYPE_Q4_0_SWZ, m_specData.nEmbd, m_specData.nFF));

        uint64_t cacheSize = 2 * m_specData.nEmbd * m_maxCacheEntries;
        OFFSET(m_layerOffsets, cacheKeys, cacheSize);
        OFFSET(m_layerOffsets, cacheValues, cacheSize);

        OFFSET(m_globalOffsets, constants[0], sizeof(shader::GlobalConstantBuffer));
        OFFSET(m_globalOffsets, constants[1], sizeof(shader::GlobalConstantBuffer));
        OFFSET(m_globalOffsets, historyIndex, 4 * 2048);

        uint64_t embdMatrixSize = vktypeSize(VKTYPE_Q4_0_LINEAR, m_specData.nEmbd, m_numVocab);
        OFFSET(m_globalOffsets, embedding, embdMatrixSize);

        OFFSET(m_globalOffsets, bypass, 2 * m_specData.nEmbd);
        OFFSET(m_globalOffsets, stage1, 2 * m_specData.nEmbd);
        OFFSET(m_globalOffsets, stage2, 2 * m_specData.nFF); // can hold both nEmbd and nFF

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

        OFFSET(m_hostOffsets, debugActivations, 2 * m_specData.nFF);
        m_hostMemory = device.allocateHost(m_hostOffsets.size);
        if (!m_hostMemory.valid())
            exit(1);
    }

    // Step 3: Set up descriptor sets
    {
        unsigned globalSets = sizeof(m_dsetGlobal) / sizeof(m_dsetGlobal[0]); // Double buffer
        unsigned layerSets = m_numLayers;

        const VkDescriptorPoolSize poolSizes[] = {
            {
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    globalSets * 1,
            },
            {
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    (uint32_t)(
                        globalSets * (sizeof(g_dsetLayoutGlobal) / sizeof(g_dsetLayoutGlobal[0]) - 1)
                        + layerSets * sizeof(g_dsetLayoutPerLayer) / sizeof(g_dsetLayoutPerLayer[0])
                    ),
            },
        };

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets = globalSets + layerSets;
        descriptorPoolCreateInfo.poolSizeCount = sizeof(poolSizes) / sizeof(poolSizes[0]);
        descriptorPoolCreateInfo.pPoolSizes = poolSizes;
        LLVK_CHECK_RESULT(vk.CreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &m_descriptorPool));

        device.allocateDescriptorSets(m_descriptorPool, m_dsetLayoutGlobal, globalSets, m_dsetGlobal);
        m_dsetLayer.resize(m_numLayers);
        device.allocateDescriptorSets(m_descriptorPool, m_dsetLayoutPerLayer, m_numLayers, m_dsetLayer.data());

        WriteBufferDescriptors write(device);
        for (unsigned i = 0; i < globalSets; ++i) {
            write.writeUniform(m_dsetGlobal[i], 0, m_globalMemory.buffer(), m_globalOffsets.constants[i]);
            write.writeStorage(m_dsetGlobal[i], 1, m_globalMemory.buffer(), m_globalOffsets.historyIndex);
            write.writeStorage(m_dsetGlobal[i], 2, m_globalMemory.buffer(), m_globalOffsets.embedding);
            write.writeStorage(m_dsetGlobal[i], 3, m_globalMemory.buffer(), m_globalOffsets.bypass);
            write.writeStorage(m_dsetGlobal[i], 4, m_globalMemory.buffer(), m_globalOffsets.stage1);
            write.writeStorage(m_dsetGlobal[i], 5, m_globalMemory.buffer(), m_globalOffsets.stage2);
        }

        for (unsigned i = 0; i < m_numLayers; ++i) {
            write.writeStorage(m_dsetLayer[i], 0, m_layerMemory[i].buffer(), m_layerOffsets.attentionNorm);
            write.writeStorage(m_dsetLayer[i], 1, m_layerMemory[i].buffer(), m_layerOffsets.Wq);
            write.writeStorage(m_dsetLayer[i], 2, m_layerMemory[i].buffer(), m_layerOffsets.Wk);
            write.writeStorage(m_dsetLayer[i], 3, m_layerMemory[i].buffer(), m_layerOffsets.Wv);
            write.writeStorage(m_dsetLayer[i], 4, m_layerMemory[i].buffer(), m_layerOffsets.Wo);
            write.writeStorage(m_dsetLayer[i], 5, m_layerMemory[i].buffer(), m_layerOffsets.cacheKeys);
            write.writeStorage(m_dsetLayer[i], 6, m_layerMemory[i].buffer(), m_layerOffsets.cacheValues);
            write.writeStorage(m_dsetLayer[i], 7, m_layerMemory[i].buffer(), m_layerOffsets.ffnNorm);
            write.writeStorage(m_dsetLayer[i], 8, m_layerMemory[i].buffer(), m_layerOffsets.W1);
            write.writeStorage(m_dsetLayer[i], 9, m_layerMemory[i].buffer(), m_layerOffsets.W2);
            write.writeStorage(m_dsetLayer[i], 10, m_layerMemory[i].buffer(), m_layerOffsets.W3);
        }

        write.commit();
    }

    // Step 4: Allocate command pools
    m_computeCommandPool = device.createCommandPool(false);
    if (device.haveTransferQueue())
        m_transferCommandPool = device.createCommandPool(true);

    m_semaphore = device.createTimelineSemaphore();
}

LlamaContext::~LlamaContext() {
    destroy();
}

void LlamaContext::destroy() {
    m_semaphore.waitForever(m_numSubmits);

    vk.FreeCommandBuffers(device, m_computeCommandPool, std::min<uint64_t>(m_numSubmits, 2), m_commandBuffers);
    memset(&m_commandBuffers, 0, sizeof(m_commandBuffers));

#define DESTROY(name) \
    do { \
        if (name) { \
            vk.DestroyCommandPool(device, name, nullptr); \
            name = nullptr; \
        } \
    } while(false)

    DESTROY(m_computeCommandPool);
    DESTROY(m_transferCommandPool);

#undef DESTROY

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
    DESTROY(m_kernelThinFp16FirstRmsNorm);
    DESTROY(m_kernelThinFp16MatMulAdd);
    DESTROY(m_kernelThinFp16RmsNormAttention);
    DESTROY(m_kernelThinFp16RmsNormFFN);

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

// Tensor types in GGML files
enum {
    FTYPE_F32 = 0,
    FTYPE_F16 = 1,
    FTYPE_Q4_0 = 2,
    FTYPE_Q4_1 = 3,
};

uint64_t ftypeSize(int32_t ftype, uint64_t ne0, uint64_t ne1) {
    switch (ftype) {
    case FTYPE_F32:
        return 4 * ne0 * ne1;
    case FTYPE_F16:
        return 2 * ne0 * ne1;
    case FTYPE_Q4_0:
        assert((ne0 % 32) == 0);
        return (4 + 16) * (ne0 / 32) * ne1;
    case FTYPE_Q4_1:
        assert((ne0 % 32) == 0);
        return (2 * 4 + 16) * (ne0 / 32) * ne1;
    }
    abort();
}

enum {
    SPLIT_PARTS_COLUMN = 0,
    SPLIT_PARTS_ROW = 1,
};

struct TensorOffsets {
    int32_t ftype;
    int32_t splitType;
    int32_t numDims;
    int32_t numElementsPerPart[2];
    std::vector<size_t> offsets;
};

struct TensorUpload {
    std::string name;
    int32_t vktype;
    int32_t numElements[2];
    VkBuffer buffer;
    Range range;

    TensorUpload(std::string name, int32_t vktype, uint32_t ne0, uint32_t ne1, VkBuffer buffer, Range range)
        : name(std::move(name)), vktype(vktype)
        , numElements{(int32_t)ne0, (int32_t)ne1}
        , buffer(buffer), range(range) {}
};

struct LlamaContext::ModelUploader {
    LlamaContext &ctx;
    std::unordered_map<std::string, TensorOffsets> m_tensors;
    std::vector<std::ifstream> m_parts;
    std::vector<char> m_fileBuffer;
    TimelineSemaphore m_semaphore;
    uint64_t m_numSubmits = 0;
    size_t m_maxUploadSize = 0;
    MemoryAndBuffer m_uploadBuffer;
    VkCommandPool m_commandPool = nullptr;
    VkCommandBuffer m_commandBuffers[2] = {};
    std::vector<VkBufferMemoryBarrier2> m_queueTransfers;

    ModelUploader(LlamaContext &ctx) : ctx(ctx) {}
    ~ModelUploader() { finish(); }

    void lazyInit();
    void finish();

    void openAndScan();
    void upload(const TensorUpload *uploads, size_t numUploads);
    template <size_t N>
    void upload(const TensorUpload (&uploads)[N]) {
        upload(uploads, N);
    }
};

void LlamaContext::ModelUploader::finish() {
    if (!m_semaphore.valid())
        return;

    m_semaphore.waitForever(m_numSubmits);
    m_semaphore.reset();

    ctx.vk.FreeCommandBuffers(ctx.device, m_commandPool,
                              std::min(m_numSubmits, (uint64_t)2), m_commandBuffers);

    memset(&m_commandBuffers, 0, sizeof(m_commandBuffers));
    m_commandPool = nullptr;

    m_uploadBuffer.reset();

    m_parts.clear();
    m_tensors.clear();
    m_fileBuffer.clear();
    m_numSubmits = 0;
}

// Open the model file(s) and scan them to find the tensor offsets.
void LlamaContext::ModelUploader::openAndScan() {
    size_t perFileBufferSize = 1024 * 1024;
    m_fileBuffer.resize(ctx.m_fileInfo.n_parts * perFileBufferSize);

    for (int i = 0; i < ctx.m_fileInfo.n_parts; ++i) {
        std::string fname = ctx.m_modelPath;
        if (i > 0) {
            fname += "." + std::to_string(i);
        }

        m_parts.emplace_back(fname, std::ios::binary);
        std::ifstream &fin = m_parts.back();
        fin.rdbuf()->pubsetbuf(m_fileBuffer.data() + i * perFileBufferSize, perFileBufferSize);

        fin.seekg(0, std::ios::end);
        const size_t fileSize = fin.tellg();

        fin.seekg(ctx.m_fileInfo.tensors_offset);

#define CHECK(cond) \
        do { \
            if (!(cond)) { \
                fprintf(stderr, "%s: broken model file %s: %s\n", __func__, fname.c_str(), #cond); \
                exit(1); \
            } \
        } while (false)

        for (;;) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            CHECK(n_dims <= 2);

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                CHECK(ne[i] <= std::numeric_limits<int32_t>::max() / nelements);
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            size_t offset = fin.tellg();

            TensorOffsets &tensor = m_tensors[name];
            if (tensor.offsets.empty()) {
                tensor.ftype = ftype;
                tensor.numDims = n_dims;
                tensor.numElementsPerPart[0] = ne[0];
                tensor.numElementsPerPart[1] = ne[1];

                // split_type = columns:
                // pattern:
                //   - tok_embeddings.*
                //   - layers.*.attention.wo.weight
                //   - layers.*.feed_forward.w2.weight

                // split_type = rows:
                // pattern:
                //   - output.*
                //   - layers.*.attention.wq.weight
                //   - layers.*.attention.wk.weight
                //   - layers.*.attention.wv.weight
                //   - layers.*.feed_forward.w1.weight
                //   - layers.*.feed_forward.w3.weight
                if (name.find("tok_embeddings") != std::string::npos) {
                    tensor.splitType = SPLIT_PARTS_COLUMN;
                } else if (name.find("layers") != std::string::npos) {
                    if (name.find("attention.wo.weight") != std::string::npos) {
                        tensor.splitType = SPLIT_PARTS_COLUMN;
                    } else if (name.find("feed_forward.w2.weight") != std::string::npos) {
                        tensor.splitType = SPLIT_PARTS_COLUMN;
                    } else {
                        tensor.splitType = SPLIT_PARTS_ROW;
                    }
                } else if (name.find("output") != std::string::npos) {
                    tensor.splitType = SPLIT_PARTS_ROW;
                }
            } else {
                CHECK(ftype == tensor.ftype);
                CHECK(tensor.numDims == n_dims);
                CHECK(tensor.numElementsPerPart[0] == ne[0]);
                CHECK(tensor.numElementsPerPart[1] == ne[1]);
            }

            tensor.offsets.resize(ctx.m_fileInfo.n_parts);
            CHECK(!tensor.offsets[i]);
            tensor.offsets[i] = offset;

            size_t tensorSize;
            switch (ftype) {
            case FTYPE_F32: tensorSize = 4 * size_t(nelements); break;
            case FTYPE_F16: tensorSize = 2 * size_t(nelements); break;
            case FTYPE_Q4_0:
                CHECK(ne[0] % 64 == 0); // File format block size is 32, but the GPU swizzle wants 64
                tensorSize = (4 + 16) * size_t(nelements / 32);
                break;
            case FTYPE_Q4_1:
                CHECK(ne[0] % 64 == 0); // File format block size is 32, but the GPU swizzle wants 64
                tensorSize = (2 * 4 + 16) * size_t(nelements / 32);
                break;
            default:
                CHECK(!"bad ftype");
                break;
            }

            CHECK(offset + tensorSize <= fileSize);
            fin.seekg(tensorSize, std::ios::cur);
            if (!fin) {
                fprintf(stderr, "%s: seek past tensor failed\n", __func__);
                exit(1);
            }

            if (offset + tensorSize >= fileSize)
                break;
        }

#undef CHECK
    }

    for (const auto &entry : m_tensors) {
        for (auto offset : entry.second.offsets) {
            if (!offset) {
                fprintf(stderr, "%s: missing part of tensor '%s'\n", __func__, entry.first.c_str());
                exit(1);
            }
        }
    }
}

void LlamaContext::ModelUploader::lazyInit() {
    if (m_semaphore.valid()) {
        assert(m_uploadBuffer.valid());
        return;
    }

    assert(!m_uploadBuffer.valid());

    m_semaphore = ctx.device.createTimelineSemaphore();

    size_t firstUploadSize = alignPot(ctx.m_globalOffsets.embedding.range, 256);
    size_t layerUploadSize =
        alignPot(ctx.m_layerOffsets.attentionNorm.range, 256) +
        alignPot(ctx.m_layerOffsets.Wq.range, 256) +
        alignPot(ctx.m_layerOffsets.Wk.range, 256) +
        alignPot(ctx.m_layerOffsets.Wv.range, 256) +
        alignPot(ctx.m_layerOffsets.Wo.range, 256) +
        alignPot(ctx.m_layerOffsets.ffnNorm.range, 256) +
        alignPot(ctx.m_layerOffsets.W1.range, 256) +
        alignPot(ctx.m_layerOffsets.W2.range, 256) +
        alignPot(ctx.m_layerOffsets.W3.range, 256);

    m_maxUploadSize = std::max(firstUploadSize, layerUploadSize);

    size_t uploadBufferSize = 2 * m_maxUploadSize;
    m_uploadBuffer = ctx.device.allocateHost(uploadBufferSize);

    m_commandPool = ctx.m_transferCommandPool;
    if (!m_commandPool)
        m_commandPool = ctx.m_computeCommandPool;
}

void LlamaContext::ModelUploader::upload(const TensorUpload *uploads, size_t numUploads) {
    lazyInit();

    // Wait for the second-to-last submit to complete. This protects against
    // a WAR hazard when writing to our host-side upload buffer and re-using
    // the command buffer.
    if (m_numSubmits >= 2)
        m_semaphore.waitForever(m_numSubmits - 1);

    // Step 1: Allocate the upload command buffer
    VkCommandBuffer cmdBuf;
    {
        if (m_numSubmits < 2) {
            VkCommandBufferAllocateInfo allocateInfo = {};
            allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocateInfo.commandPool = m_commandPool;
            allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocateInfo.commandBufferCount = 1;

            LLVK_CHECK_RESULT(ctx.vk.AllocateCommandBuffers(ctx.device, &allocateInfo, &cmdBuf));
            m_commandBuffers[m_numSubmits] = cmdBuf;
        } else {
            // The wait above ensured that the old command buffer is idle, and
            // vkBeginCommandBuffer resets it implicitly.
            cmdBuf = m_commandBuffers[m_numSubmits % 2];
        }

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        LLVK_CHECK_RESULT(ctx.vk.BeginCommandBuffer(cmdBuf, &beginInfo));

        ctx.vk.utilCmdPipelineMemoryBarrier(
                cmdBuf,
                VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_WRITE_BIT,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
    }

    size_t beginQueueTransfers = m_queueTransfers.size();
    size_t base = (m_numSubmits % 2) * m_maxUploadSize;
    size_t uploadOffset = 0;

    // Step 2: Read from file(s), write to upload buffer, and build upload command buffer
    {
        void *buffer = m_uploadBuffer.map(base, m_maxUploadSize);
        VkBuffer currentDstBuffer = nullptr;
        std::vector<VkBufferCopy> copyRegions;
        std::vector<char> convertBuffer;
        auto flushCopies = [&]() {
            if (!currentDstBuffer)
                return;

            ctx.vk.CmdCopyBuffer(cmdBuf, m_uploadBuffer.buffer(), currentDstBuffer,
                                 copyRegions.size(), copyRegions.data());

            if (ctx.device.haveTransferQueue() && ctx.device.transferQueueFamily() != ctx.device.computeQueueFamily()) {
                bool isFirst = true;
                for (const VkBufferCopy &region : copyRegions) {
                    if (!isFirst) {
                        auto &prev = m_queueTransfers.back();
                        if (prev.buffer == currentDstBuffer &&
                            alignPot(prev.offset + prev.size, 256) == region.dstOffset)
                        {
                            prev.size += region.size;
                            continue;
                        }
                    }
                    isFirst = false;

                    VkBufferMemoryBarrier2 queueTransfer = {};
                    queueTransfer.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                    queueTransfer.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                    queueTransfer.srcAccessMask = 0;
                    queueTransfer.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    queueTransfer.dstAccessMask = 0;
                    queueTransfer.srcQueueFamilyIndex = ctx.device.transferQueueFamily();
                    queueTransfer.dstQueueFamilyIndex = ctx.device.computeQueueFamily();
                    queueTransfer.buffer = currentDstBuffer;
                    queueTransfer.offset = region.dstOffset;
                    queueTransfer.size = region.size;
                    m_queueTransfers.push_back(queueTransfer);
                }

                // VkDependencyInfo dependencyInfo = {};
                // dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;

                // ctx.vk.CmdPipelineBarrier2(cmdBuf, &dependencyInfo);
            }

            currentDstBuffer = nullptr;
            copyRegions.clear();
        };

        for (unsigned idx = 0; idx < numUploads; ++idx) {
            const TensorUpload &upload = uploads[idx];
            assert(uploadOffset + upload.range.range <= m_maxUploadSize);

            auto tensorIt = m_tensors.find(upload.name);
            if (tensorIt == m_tensors.end()) {
                fprintf(stderr, "%s: missing tensor '%s'\n", __func__, upload.name.c_str());
                exit(1);
            }
            const TensorOffsets &tensor = tensorIt->second;

            // Per-tensor sanity checks
            const char *typeCheckFail = nullptr;
            switch (upload.vktype) {
            case VKTYPE_F16:
                if (tensor.ftype != FTYPE_F32 && tensor.ftype != FTYPE_F16)
                    typeCheckFail = "f16";
                break;
            case VKTYPE_Q4_0_SWZ:
            case VKTYPE_Q4_0_LINEAR:
                if (tensor.ftype != FTYPE_Q4_0)
                    typeCheckFail = "q4_0";
                break;
            default:
                typeCheckFail = "unknown";
                break;
            }
            if (typeCheckFail) {
                fprintf(stderr, "%s: upload '%s' as %s, but ftype is %u\n",
                        __func__, upload.name.c_str(), typeCheckFail, tensor.ftype);
                exit(1);
            }

            int32_t fullTensorElements[2] = { tensor.numElementsPerPart[0], tensor.numElementsPerPart[1] };

            if (tensor.numDims == 2) {
                if (tensor.splitType == SPLIT_PARTS_COLUMN) {
                    fullTensorElements[0] *= m_parts.size();
                } else {
                    fullTensorElements[1] *= m_parts.size();
                }
            }

            if (fullTensorElements[0] != upload.numElements[0] ||
                fullTensorElements[1] != upload.numElements[1])
            {
                fprintf(stderr, "%s: tensor '%s' has size %dx%x (per part: %dx%d), expect %dx%d\n",
                        __func__, upload.name.c_str(),
                        fullTensorElements[0], fullTensorElements[1],
                        tensor.numElementsPerPart[0], tensor.numElementsPerPart[1],
                        upload.numElements[0], upload.numElements[1]);
                exit(1);
            }

            // Load from file and copy/swizzle to upload buffer.
            size_t tensorParts = tensor.numDims == 2 ? m_parts.size() : 1;
            size_t srcRowBytes = ftypeSize(tensor.ftype, tensor.numElementsPerPart[0], 1);

            assert(vktypeSize(upload.vktype, fullTensorElements[0], fullTensorElements[1]) <= upload.range.range);

            if ((upload.vktype == VKTYPE_F16 && tensor.ftype == FTYPE_F32) ||
                upload.vktype == VKTYPE_Q4_0_SWZ || upload.vktype == VKTYPE_Q4_0_LINEAR) {
                convertBuffer.resize(srcRowBytes);
            }

            char * __restrict__ cbuf = convertBuffer.data();

            for (unsigned part = 0; part < tensorParts; ++part) {
                auto &fin = m_parts[part];
                fin.seekg(tensor.offsets[part], std::ios::beg);

                if (upload.vktype == VKTYPE_Q4_0_SWZ || upload.vktype == VKTYPE_Q4_0_LINEAR) {
                    assert(tensor.numElementsPerPart[0] % 64 == 0);
                    size_t nDoubleBlocks = size_t(fullTensorElements[0] / 64);
                    size_t dstScaleBytes = 4 * nDoubleBlocks * fullTensorElements[1];
                    size_t dstWeightsBytes = 32 * nDoubleBlocks * fullTensorElements[1];
                    char * __restrict__ tensorUploadScales = (char *)buffer + uploadOffset;
                    char * __restrict__ tensorUploadWeights = (char *)buffer + uploadOffset + dstScaleBytes;
                    size_t scaleRowStride;
                    size_t weightsRowStride;
                    size_t scaleBlockStride;
                    size_t weightsBlockStride;

                    if (upload.vktype == VKTYPE_Q4_0_SWZ) {
                        scaleRowStride = 4;
                        weightsRowStride = 32;
                        scaleBlockStride = 4 * fullTensorElements[1];
                        weightsBlockStride = 32 * fullTensorElements[1];
                    } else {
                        scaleRowStride = 4 * nDoubleBlocks;
                        weightsRowStride = 32 * nDoubleBlocks;
                        scaleBlockStride = 4;
                        weightsBlockStride = 32;
                    }

                    if (tensor.splitType == SPLIT_PARTS_COLUMN) {
                        tensorUploadScales += part * dstScaleBytes / tensorParts;
                        tensorUploadWeights += part * dstWeightsBytes / tensorParts;
                    } else {
                        tensorUploadScales += part * 4 * fullTensorElements[1] / tensorParts;
                        tensorUploadWeights += part * 32 * fullTensorElements[1] / tensorParts;
                    }

                    for (unsigned row = 0; row < tensor.numElementsPerPart[1]; ++row) {
                        fin.read(cbuf, srcRowBytes);

                        for (unsigned block = 0; block < nDoubleBlocks / tensorParts; ++block) {
                            size_t scaleOffset = row * scaleRowStride + block * scaleBlockStride;
                            size_t weightsOffset = row * weightsRowStride + block * weightsBlockStride;
                            *(_Float16 *)(tensorUploadScales + scaleOffset + 0) = *(float *)(cbuf + block * 40 + 0);
                            *(_Float16 *)(tensorUploadScales + scaleOffset + 2) = *(float *)(cbuf + block * 40 + 20);
                            memcpy(tensorUploadWeights + weightsOffset +  0, cbuf + block * 40 + 4, 16);
                            memcpy(tensorUploadWeights + weightsOffset + 16, cbuf + block * 40 + 24, 16);

                            if (row == 0 && block < 2) {
                                if (upload.name == "layers.0.attention.wq.weight") {
                                    printf("Upload %s, part %u row %u block %u\n", upload.name.c_str(), part, row, block);
                                    printf("  scales: %f %08x %f %08x\n",
                                        *(float *)(cbuf + block * 40 + 0), *(int *)(cbuf + block * 40 + 0),
                                        *(float *)(cbuf + block * 40 + 20), *(int *)(cbuf + block * 40 + 20));
                                    printf("  converted: %08x\n", *(int *)(tensorUploadScales + scaleOffset));
                                    int *w = (int *)(tensorUploadWeights + weightsOffset);
                                    printf("  weights: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                                        w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
                                }
                            }
                        }
                    }
                } else {
                    char * __restrict__ tensorUpload = (char *)buffer + uploadOffset;
                    size_t dstRowBytes = vktypeSize(upload.vktype, fullTensorElements[0], 1);
                    if (tensor.splitType == SPLIT_PARTS_COLUMN) {
                        tensorUpload += part * (dstRowBytes / tensorParts);
                    } else {
                        tensorUpload += part * dstRowBytes * (fullTensorElements[1] / tensorParts);
                    }

                    assert(upload.vktype == VKTYPE_F16);

                    switch (tensor.ftype) {
                    case FTYPE_F16:
                        for (unsigned row = 0; row < tensor.numElementsPerPart[1]; ++row) {
                            char *rowUpload = tensorUpload + row * dstRowBytes;
                            fin.read(rowUpload, srcRowBytes);
                        }
                        break;
                    case FTYPE_F32:
                        for (unsigned row = 0; row < tensor.numElementsPerPart[1]; ++row) {
                            fin.read(cbuf, srcRowBytes);
                            for (unsigned col = 0; col < tensor.numElementsPerPart[0]; ++col) {
                                *(_Float16 *)(tensorUpload + row * dstRowBytes + 2 * col) = *(float *)(cbuf + 4 * col);
                            }
                        }
                        break;
                    default:
                        abort();
                    }
                }
            }

            // Setup the copy command.
            if (currentDstBuffer && currentDstBuffer != upload.buffer)
                flushCopies();
            currentDstBuffer = upload.buffer;

            VkBufferCopy region;
            region.srcOffset = base + uploadOffset;
            region.dstOffset = upload.range.offset;
            region.size = upload.range.range;
            copyRegions.push_back(region);

            uploadOffset += alignPot(upload.range.range, 256);
        }

        flushCopies();

        m_uploadBuffer.unmap();

        VkDependencyInfo dependencyInfo = {};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.bufferMemoryBarrierCount = m_queueTransfers.size() - beginQueueTransfers;
        dependencyInfo.pBufferMemoryBarriers = m_queueTransfers.data() + beginQueueTransfers;
        ctx.vk.CmdPipelineBarrier2(cmdBuf, &dependencyInfo);

        LLVK_CHECK_RESULT(ctx.vk.EndCommandBuffer(cmdBuf));
    }

    // Step 3: Submit the upload command buffer
    m_numSubmits++;

    VkQueue queue = ctx.device.haveTransferQueue() ? ctx.device.transferQueue() : ctx.device.computeQueue();

    VkCommandBufferSubmitInfo commandBufferSubmitInfo = {};
    commandBufferSubmitInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferSubmitInfo.commandBuffer = cmdBuf;

    VkSemaphoreSubmitInfo semaphoreSubmitInfo = {};
    semaphoreSubmitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    semaphoreSubmitInfo.semaphore = m_semaphore.semaphore();
    semaphoreSubmitInfo.value = m_numSubmits;
    semaphoreSubmitInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkSubmitInfo2 submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &semaphoreSubmitInfo;

    LLVK_CHECK_RESULT(ctx.vk.QueueSubmit2(queue, 1, &submitInfo, nullptr));
}

void LlamaContext::uploadModel() {
    fprintf(stderr, "llama-vk: uploading");
    fflush(stderr);

    m_uploader = std::make_unique<ModelUploader>(*this);
    m_uploader->openAndScan();

    // Upload tensors.
    //
    // PCIe is relatively slow, so pipeline the upload.
    //
    // Swizzling is currently done on the CPU. There's space for experimentation
    // about what strategy is best with batched prompt processing: async compute
    // fetching directly from mmap()'d buffers via VK_EXT_external_memory_host,
    // in parallel with prompt processing, would be interesting.
    {
        TensorUpload uploads[] = {
            {"tok_embeddings.weight", VKTYPE_Q4_0_LINEAR, m_specData.nEmbd, m_numVocab,
             m_globalMemory.buffer(), m_globalOffsets.embedding},
        };
        m_uploader->upload(uploads);
        fprintf(stderr, ".");
        fflush(stderr);
    }

    for (unsigned layer = 0; layer < 1 /*m_numLayers*/; ++layer) {
        TensorUpload uploads[] = {
            {"layers." + std::to_string(layer) + ".attention_norm.weight", VKTYPE_F16, m_specData.nEmbd, 1,
             m_layerMemory[layer].buffer(), m_layerOffsets.attentionNorm},
            {"layers." + std::to_string(layer) + ".attention.wq.weight", VKTYPE_Q4_0_SWZ, m_specData.nEmbd, m_specData.nEmbd,
             m_layerMemory[layer].buffer(), m_layerOffsets.Wq},
            {"layers." + std::to_string(layer) + ".attention.wk.weight", VKTYPE_Q4_0_SWZ, m_specData.nEmbd, m_specData.nEmbd,
             m_layerMemory[layer].buffer(), m_layerOffsets.Wk},
            {"layers." + std::to_string(layer) + ".attention.wv.weight", VKTYPE_Q4_0_SWZ, m_specData.nEmbd, m_specData.nEmbd,
             m_layerMemory[layer].buffer(), m_layerOffsets.Wv},
            {"layers." + std::to_string(layer) + ".attention.wo.weight", VKTYPE_Q4_0_SWZ, m_specData.nEmbd, m_specData.nEmbd,
             m_layerMemory[layer].buffer(), m_layerOffsets.Wo},
            {"layers." + std::to_string(layer) + ".ffn_norm.weight", VKTYPE_F16, m_specData.nEmbd, 1,
             m_layerMemory[layer].buffer(), m_layerOffsets.ffnNorm},
            {"layers." + std::to_string(layer) + ".feed_forward.w1.weight", VKTYPE_Q4_0_SWZ, m_specData.nEmbd, m_specData.nFF,
             m_layerMemory[layer].buffer(), m_layerOffsets.W1},
            {"layers." + std::to_string(layer) + ".feed_forward.w2.weight", VKTYPE_Q4_0_SWZ, m_specData.nFF, m_specData.nEmbd,
             m_layerMemory[layer].buffer(), m_layerOffsets.W2},
            {"layers." + std::to_string(layer) + ".feed_forward.w3.weight", VKTYPE_Q4_0_SWZ, m_specData.nEmbd, m_specData.nFF,
             m_layerMemory[layer].buffer(), m_layerOffsets.W3},
        };
        m_uploader->upload(uploads);
        fprintf(stderr, ".");
        fflush(stderr);
    }

    m_uploader->finish();
    fprintf(stderr, "\n");
    fflush(stderr);
}

void LlamaContext::process(const llama_token *tokens, size_t numTokens) {
    if (m_numSubmits >= 2)
        m_semaphore.waitForever(m_numSubmits - 1);

    // Step 1: Obtain command buffer and initialize constants.
    VkCommandBuffer cmdBuf;
    {
        if (m_numSubmits < 2) {
            VkCommandBufferAllocateInfo allocateInfo = {};
            allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocateInfo.commandPool = m_computeCommandPool;
            allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocateInfo.commandBufferCount = 1;

            LLVK_CHECK_RESULT(vk.AllocateCommandBuffers(device, &allocateInfo, &cmdBuf));
            m_commandBuffers[m_numSubmits] = cmdBuf;
        } else {
            // The wait above ensured that the old command buffer is idle, and
            // vkBeginCommandBuffer resets it implicitly.
            cmdBuf = m_commandBuffers[m_numSubmits % 2];
        }

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        LLVK_CHECK_RESULT(vk.BeginCommandBuffer(cmdBuf, &beginInfo));

        if (m_uploader && !m_uploader->m_queueTransfers.empty()) {
            VkDependencyInfo dependencyInfo = {};
            dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependencyInfo.bufferMemoryBarrierCount = m_uploader->m_queueTransfers.size();
            dependencyInfo.pBufferMemoryBarriers = m_uploader->m_queueTransfers.data();
            vk.CmdPipelineBarrier2(cmdBuf, &dependencyInfo);

            m_uploader->m_queueTransfers.clear();
        }
    }

    shader::GlobalConstantBuffer constants = {};
    constants.rmsEpsilon = 1e-6f;
    constants.currentRotaryPosition = 0;
    constants.currentHistoryBase = 0;
    constants.currentHistoryLength = 0;
    constants.currentStorageIndex = 0;
    constants.numKeyValueEntries = m_maxCacheEntries;
    constants.currentToken = tokens[0];

    vk.CmdUpdateBuffer(cmdBuf, m_globalMemory.buffer(), m_globalOffsets.constants[m_numPasses % 2].offset,
                       sizeof(constants), &constants);

    vk.utilCmdPipelineMemoryBarrier(cmdBuf,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

    // Step 2: Inference
    const VkDescriptorSet sets[] = {
        m_dsetGlobal[m_numPasses % 2],
        m_dsetLayer[0],
    };
    vk.CmdBindDescriptorSets(
            cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout,
            0, sizeof(sets) / sizeof(sets[0]), sets, 0, nullptr);
    vk.CmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_kernelThinFp16FirstRmsNorm);
    vk.CmdDispatch(cmdBuf, 1, 1, 1);

    vk.utilCmdPipelineMemoryBarrier(cmdBuf,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    vk.CmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_kernelThinFp16Attention);
    vk.CmdDispatch(cmdBuf, m_specData.nHead, 1, 1);

    vk.utilCmdPipelineMemoryBarrier(cmdBuf,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    vk.CmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_kernelThinFp16MatMulAdd);
    vk.CmdDispatch(cmdBuf, m_specData.nEmbd / NUM_THIN_MATMUL_THREADS, 1, 1);

    vk.utilCmdPipelineMemoryBarrier(cmdBuf,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    vk.CmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_kernelThinFp16RmsNormFFN);
    vk.CmdDispatch(cmdBuf, 1, 1, 1);

    vk.utilCmdPipelineMemoryBarrier(cmdBuf,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

    {
        VkBufferCopy region;
        region.srcOffset = m_globalOffsets.stage1.offset;
        region.dstOffset = m_hostOffsets.debugActivations.offset;
        region.size = 2 * m_specData.nEmbd;

        vk.CmdCopyBuffer(cmdBuf, m_globalMemory.buffer(), m_hostMemory.buffer(), 1, &region);
    }

    vk.utilCmdPipelineMemoryBarrier(cmdBuf,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);

    // Step 3: Submit the upload command buffer
    vk.EndCommandBuffer(cmdBuf);

    m_numSubmits++;

    VkCommandBufferSubmitInfo commandBufferSubmitInfo = {};
    commandBufferSubmitInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferSubmitInfo.commandBuffer = cmdBuf;

    VkSemaphoreSubmitInfo semaphoreSubmitInfo = {};
    semaphoreSubmitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    semaphoreSubmitInfo.semaphore = m_semaphore.semaphore();
    semaphoreSubmitInfo.value = m_numSubmits;
    semaphoreSubmitInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkSubmitInfo2 submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &semaphoreSubmitInfo;

    LLVK_CHECK_RESULT(vk.QueueSubmit2(device.computeQueue(), 1, &submitInfo, nullptr));

    // Step 4: Dump debug activations.
    {
        m_semaphore.waitForever(m_numSubmits);

        void *activations =
                m_hostMemory.map(m_hostOffsets.debugActivations.offset,
                                 m_hostOffsets.debugActivations.range);

        fprintf(stderr, "Debug activations:\n");
        for (unsigned i = 0; i < 128; ++i) {
            float val = *((_Float16 *)activations + i);
            fprintf(stderr, " %+8f", val);
            if (i % 8 == 7)
                fprintf(stderr, "\n");
        }

        m_hostMemory.unmap();
    }
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

    vkctx.uploadModel();

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // Tokenize the prompt
    auto embd_inp = llama_tokenize(ctx, params.prompt.c_str(), true);

    printf("Initial embd_inp:\n");
    for (const auto &token : embd_inp)
        printf("  %u: '%s'\n", token, llama_token_to_str(ctx, token));
    printf("--\n");

    vkctx.process(embd_inp.data(), embd_inp.size());

    return 0;
}
