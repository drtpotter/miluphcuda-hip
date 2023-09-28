# This is the miluphcuda Makefile.

CC = g++

CFLAGS   = -c -std=c99 -O3 -DVERSION=\"$(GIT_VERSION)\" -fPIC
#CFLAGS   = -c -std=c99 -g
LDFLAGS  = -lm

GIT_VERSION := $(shell git describe --abbrev=4 --dirty --always --tags)

HIP_DIR = $(shell hipconfig -p)

HIPCC = ${HIP_DIR}/bin/hipcc 

# for using a debugger use the first, otherwise the third:
#HIPFLAGS  = -ccbin ${CC} -x cu -c -dc  -G -lineinfo  -Xcompiler "-rdynamic -g -pthread"  -DVERSION=\"$(GIT_VERSION)\"
#HIPFLAGS  = -x cu -c -dc -O3  -Xcompiler "-O3 -pthread" --ptxas-options=-v
HIPFLAGS  = -fgpu-rdc -c -O3 -DVERSION=\"$(GIT_VERSION)\" 

HIP_LIB      = ${HIP_DIR}
INCLUDE_DIRS += -I$(HIP_LIB)/include -I/usr/include/hdf5/serial -I/usr/lib/openmpi/include/
# if you use HDF5 I/O use the first, otherwise the second:
LDFLAGS      += -L$(HIP_LIB)/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lpthread -lconfig -lhdf5
#LDFLAGS      += -ccbin ${CC} -L$(HIP_LIB)/lib64 -lcudart -lpthread -lconfig


# default target
all: miluphcuda

# headers and object files
HEADERS =
HIP_HEADERS =  cuda_utils.h  checks.h io.h  miluph.h  parameter.h  timeintegration.h  tree.h  euler.h rk2adaptive.h pressure.h soundspeed.h device_tools.h boundary.h predictor_corrector.h predictor_corrector_euler.h memory_handling.h plasticity.h porosity.h aneos.h kernel.h linalg.h xsph.h density.h rhs.h internal_forces.h velocity.h damage.h little_helpers.h gravity.h viscosity.h artificial_stress.h stress.h extrema.h sinking.h coupled_heun_rk4_sph_nbody.h rk4_pointmass.h config_parameter.h
OBJ =
HIP_OBJ = io.o  miluph.o  boundary.o timeintegration.o tree.o memory_handling.o euler.o rk2adaptive.o pressure.o soundspeed.o device_tools.o predictor_corrector.o predictor_corrector_euler.o plasticity.o porosity.o aneos.o kernel.o linalg.o xsph.o density.o rhs.o internal_forces.o velocity.o damage.o little_helpers.o gravity.o viscosity.o artificial_stress.o stress.o extrema.o sinking.o coupled_heun_rk4_sph_nbody.o rk4_pointmass.o config_parameter.o


documentation:
	cd doc && make all > .log

miluphcuda: $(OBJ) $(HIP_OBJ)
#	$(HIPCC) $(GPU_ARCH) $(HIP_LINK_FLAGS) -o $(HIP_LINK_OBJ) $(HIP_OBJ)
	$(HIPCC) -fgpu-rdc $(HIP_OBJ) $(LDFLAGS) -o $@
#	$(CC) $(OBJ) $(HIP_OBJ) $(HIP_LINK_OBJ) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) $<

%.o: %.cu
	$(HIPCC) $(GPU_ARCH) $(HIPFLAGS) $(INCLUDE_DIRS) $<

.PHONY: clean
clean:
	@rm -f	*.o miluphcuda
	@echo make clean: done


# dependencies for object files
$(OBJ):  $(HEADERS) Makefile
$(HIP_OBJ): $(HEADERS) $(HIP_HEADERS) Makefile
