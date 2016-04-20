#This is a general OpenACC and CUDA makefile
#edit the paths to cudalib
#setup to use cce. If using gcc, undefine __USE_ACC until openacc is supported

CXX=CC
LD=CC


CXX_FLAGS= -O2 -hstd=c++11 -hlist=am -static #-hflex_mp=intolerant#-hfp4 -haggress -hcache3 -hthread3 -hflex_mp=tolerant -hipa4 -hautothread #-hsystem_alloc -hnomemory_alloc_intrinsics -hdep_name -hlist=am
LD_FLAGS= -hstd=c++11 -homp

INCLUDE= -I.#-D__USE_ACC

SRCS_CXX=DeepNet.cpp main.cpp
OBJS_CXX=$(subst .cpp,.o,$(SRCS_CXX))


all: MAIN

MAIN: $(SRCS_CXX)
	$(CXX) $(CXX_FLAGS) $(INCLUDE) $(SRCS_CXX) -o DeepRBM.x

LINK: $(OBJS_CXX)
	$(LD) $(LD_FLAGS) $(INCLUDE) $(OBJS_CXX) $(OBJS_ACC_CXX) -o DeepRBM.x

depend: .depend

.depend: $(OBJS_CXX) $(OBJS_ACC_CXX)
	rm -f ./.depend
	$(CXX) $(CXXFLAGS) -MM $^>>./.depend;


MISC_REM = *.o *.ptx *.cub

clean: 
	$(RM) $(OBJS_CXX) $(OBJS_ACC_CXX) $(OBJS_CURAND) $(MISC_REM)

dist-clean: clean
	$(RM) *~ .dependtool

