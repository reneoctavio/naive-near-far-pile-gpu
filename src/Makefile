ifeq ($(dbg),1)
	NVCCFLAGS += -g -G
endif

ifdef NVCC_BITS
	NVCCFLAGS += -m $(NVCC_BITS)
endif

ifdef NVCC_VERBOSE
	NVCCFLAGS += -Xptxas="-v"
endif

INCLUDES := -I ./include/

GENCODE_SM20	:= -gencode arch=compute_20,code=sm_20
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52

GENCODE_FLAGS	:= $(GENCODE_SM52)

NVCCFLAGS += $(GENCODE_FLAGS) $(INCLUDES)
NVCCFLAGS += -Xcompiler -static-libgcc -Xcompiler -static-libstdc++

all: build

build: main

sssp.o: sssp.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

main.o: main.cu 
	nvcc $(NVCCFLAGS) -o $@ -c $<

main: main.o sssp.o 
	nvcc $(NVCCFLAGS) -o $@ $+

run: build
	./main

clean:
	rm -f main.o sssp.o main

clobber:
	clean
