ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
ifeq ($(UNAME_S),Darwin)
	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
		ifeq ($(SYSCTL_M),1)
			# UNAME_P := arm
			# UNAME_M := arm64
			warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
		endif
	endif
endif

#
# Compile flags
#

# keep standard at C11 and C++17
CFLAGS   = -I.              -std=c11   -fPIC
CXXFLAGS = -I. -I./examples -std=c++17 -fPIC
LDFLAGS  =

ifdef LLAMA_DEBUG
CFLAGS   += -g -Og
CXXFLAGS += -g -Og
else
CFLAGS   += -O3 -DNDEBUG
CXXFLAGS += -O3 -DNDEBUG
endif

# warnings
CFLAGS   += -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function
CXXFLAGS += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar

# OS specific
# TODO: support Windows
ifeq ($(UNAME_S),Linux)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Darwin)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),FreeBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),NetBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),OpenBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Haiku)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686))
	# Use all CPU extensions that are available:
	CFLAGS += -march=native -mtune=native
	CXXFLAGS += -march=native -mtune=native
endif
ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		CFLAGS += -mcpu=power9
		CXXFLAGS += -mcpu=power9
	endif
	# Require c++23's std::byteswap for big-endian support.
	ifeq ($(UNAME_M),ppc64)
		CXXFLAGS += -std=c++23 -DGGML_BIG_ENDIAN
	endif
endif
ifndef LLAMA_NO_ACCELERATE
	# Mac M1 - include Accelerate framework.
	# `-framework Accelerate` works on Mac Intel as well, with negliable performance boost (as of the predict time).
	ifeq ($(UNAME_S),Darwin)
		CFLAGS  += -DGGML_USE_ACCELERATE
		LDFLAGS += -framework Accelerate
	endif
endif
ifdef LLAMA_OPENBLAS
	CFLAGS  += -DGGML_USE_OPENBLAS -I/usr/local/include/openblas
	LDFLAGS += -lopenblas
endif
ifdef LLAMA_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif
ifneq ($(filter aarch64%,$(UNAME_M)),)
	CFLAGS += -mcpu=native
	CXXFLAGS += -mcpu=native
endif
ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, 2, 3
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif
ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 4
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif
ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 4
	CFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif

#
# Print build information
#

$(info I llama.cpp build info: )
$(info I UNAME_S:  $(UNAME_S))
$(info I UNAME_P:  $(UNAME_P))
$(info I UNAME_M:  $(UNAME_M))
$(info I CFLAGS:   $(CFLAGS))
$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS:  $(LDFLAGS))
$(info I CC:       $(CCV))
$(info I CXX:      $(CXXV))
$(info )

default: main quantize perplexity embedding llama-vk

#
# Build library
#

ggml.o: ggml.c ggml.h
	$(CC)  $(CFLAGS)   -c ggml.c -o ggml.o

llama.o: llama.cpp llama.h llama_util.h llama_internal.h
	$(CXX) $(CXXFLAGS) -c llama.cpp -o llama.o

common.o: examples/common.cpp examples/common.h
	$(CXX) $(CXXFLAGS) -c examples/common.cpp -o common.o

clean:
	rm -vf *.o main quantize quantize-stats perplexity embedding

main: examples/main/main.cpp ggml.o llama.o common.o
	$(CXX) $(CXXFLAGS) examples/main/main.cpp ggml.o llama.o common.o -o main $(LDFLAGS)
	@echo
	@echo '====  Run ./main -h for help.  ===='
	@echo

quantize: examples/quantize/quantize.cpp ggml.o llama.o
	$(CXX) $(CXXFLAGS) examples/quantize/quantize.cpp ggml.o llama.o -o quantize $(LDFLAGS)

quantize-stats: examples/quantize-stats/quantize-stats.cpp ggml.o llama.o
	$(CXX) $(CXXFLAGS) examples/quantize-stats/quantize-stats.cpp ggml.o llama.o -o quantize-stats $(LDFLAGS)

perplexity: examples/perplexity/perplexity.cpp ggml.o llama.o common.o
	$(CXX) $(CXXFLAGS) examples/perplexity/perplexity.cpp ggml.o llama.o common.o -o perplexity $(LDFLAGS)

embedding: examples/embedding/embedding.cpp ggml.o llama.o common.o
	$(CXX) $(CXXFLAGS) examples/embedding/embedding.cpp ggml.o llama.o common.o -o embedding $(LDFLAGS)

libllama.so: llama.o ggml.o
	$(CXX) $(CXXFLAGS) -shared -fPIC -o libllama.so llama.o ggml.o $(LDFLAGS)

KERNELS := \
	KernelThinFp16Attention \
	KernelThinFp16Ffn \
	KernelThinFp16FirstRmsNorm \
	KernelThinFp16Output \
	KernelThinFp16MatMulAdd \
	KernelThinFp16RmsNorm \
	KernelUploadF32toF16 \
	KernelUploadQ4_0_linear \
	KernelUploadQ4_0_swz

llama-vk: vulkan/llama-vk.cpp ggml.o llama.o $(KERNELS:%=vulkan/%.spv)
	$(CXX) $(CXXFLAGS) vulkan/llama-vk.cpp ggml.o llama.o -lvulkan -o llama-vk

vulkan/%.spv: vulkan/llama-vk.hlsl
	dxc -spirv -T cs_6_6 -E $* -enable-16bit-types -fspv-target-env=vulkan1.3 -Fo $@ $<

#
# Tests
#

.PHONY: tests
tests:
	bash ./tests/run-tests.sh
