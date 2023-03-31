
#include <array>
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char **argv) {
    PFN_vkEnumerateInstanceLayerProperties EnumerateInstanceLayerProperties = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceLayerProperties"));
    PFN_vkCreateInstance CreateInstance = reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr(NULL, "vkCreateInstance"));

    uint32_t layerCount;
    std::vector<VkLayerProperties> layerProperties;
    LLVK_CHECK_RESULT(EnumerateInstanceLayerProperties(&layerCount, NULL));
    layerProperties.resize(layerCount);
    LLVK_CHECK_RESULT(EnumerateInstanceLayerProperties(&layerCount, layerProperties.data()));

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



    PFN_vkDestroyInstance DestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(vkGetInstanceProcAddr(instance, "vkDestroyInstance"));
    DestroyInstance(instance, NULL);

    printf("Hi\n");
    return 0;
}
