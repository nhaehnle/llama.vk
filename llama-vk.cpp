
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

        if (properties.properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 3, 0)) {
            printf("   --- ignoring device, require Vulkan 1.3\n");
            continue;
        }

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

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
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

} // namespace llvk

int main(int argc, char **argv) {
    auto vk = llvk::OwnedInstance::createDefault();

    auto device = llvk::OwnedDevice::createDefault(vk);


    printf("Hi rebuild\n");
    return 0;
}
