TRUNK_DIR=../..

SRC_DIR=$(TRUNK_DIR)/src
SIMD_SRC_DIR=$(SRC_DIR)/Simd
TEST_SRC_DIR=$(SRC_DIR)/Test

CXX=g++ -I$(SRC_DIR) -fPIC
LNK=g++
DIR_GUARD=@mkdir -p $(@D)

ifeq ($(DEBUG),1)
CFG=Debug
CXX+= -g -DDEBUG
else
CFG=Release
CXX+= -O3 -DNDEBUG
endif

OBJ_DIR=$(TRUNK_DIR)/obj/$(CFG)
SIMD_OBJ_DIR=$(OBJ_DIR)/Simd
TEST_OBJ_DIR=$(OBJ_DIR)/Test

BIN_DIR=$(TRUNK_DIR)/bin/$(CFG)
LNK+=-L$(BIN_DIR)

SIMD_DEPS = $(shell ls $(SIMD_SRC_DIR)/*.h)

SIMD_BASE_SRCS = $(shell ls $(SIMD_SRC_DIR)/SimdBase*.cpp)
SIMD_BASE_OBJS = $(patsubst $(SIMD_SRC_DIR)/%.cpp, $(SIMD_OBJ_DIR)/%.o, $(SIMD_BASE_SRCS))
SIMD_BASE_LIB=$(BIN_DIR)/libSimdBase.a

$(SIMD_BASE_LIB): $(SIMD_BASE_OBJS)
	$(DIR_GUARD) 
	$(LNK) -shared -static $(SIMD_BASE_OBJS) -o $@

$(SIMD_OBJ_DIR)/SimdBase%.o: $(SIMD_SRC_DIR)/SimdBase%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -c -o $@ $< 

SIMD_SSE2_SRCS = $(shell ls $(SIMD_SRC_DIR)/SimdSse2*.cpp)
SIMD_SSE2_OBJS = $(patsubst $(SIMD_SRC_DIR)/%.cpp, $(SIMD_OBJ_DIR)/%.o, $(SIMD_SSE2_SRCS))
SIMD_SSE2_LIB=$(BIN_DIR)/libSimdSse2.a

$(SIMD_SSE2_LIB): $(SIMD_SSE2_OBJS)
	$(DIR_GUARD) 
	$(LNK) -shared -static $(SIMD_SSE2_OBJS) -o $@

$(SIMD_OBJ_DIR)/SimdSse2%.o: $(SIMD_SRC_DIR)/SimdSse2%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -march=pentium4 -c -o $@ $< 

SIMD_SSSE3_SRCS = $(shell ls $(SIMD_SRC_DIR)/SimdSsse3*.cpp)
SIMD_SSSE3_OBJS = $(patsubst $(SIMD_SRC_DIR)/%.cpp, $(SIMD_OBJ_DIR)/%.o, $(SIMD_SSSE3_SRCS))
SIMD_SSSE3_LIB=$(BIN_DIR)/libSimdSsse3.a

$(SIMD_SSSE3_LIB): $(SIMD_SSSE3_OBJS)
	$(DIR_GUARD) 
	$(LNK) -shared -static $(SIMD_SSSE3_OBJS) -o $@

$(SIMD_OBJ_DIR)/SimdSsse3%.o: $(SIMD_SRC_DIR)/SimdSsse3%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -march=core2 -c -o $@ $< 

SIMD_SSE41_SRCS = $(shell ls $(SIMD_SRC_DIR)/SimdSse41*.cpp)
SIMD_SSE41_OBJS = $(patsubst $(SIMD_SRC_DIR)/%.cpp, $(SIMD_OBJ_DIR)/%.o, $(SIMD_SSE41_SRCS))
SIMD_SSE41_LIB=$(BIN_DIR)/libSimdSse41.a

$(SIMD_SSE41_LIB): $(SIMD_SSE41_OBJS)
	$(DIR_GUARD) 
	$(LNK) -shared -static $(SIMD_SSE41_OBJS) -o $@

$(SIMD_OBJ_DIR)/SimdSse41%.o: $(SIMD_SRC_DIR)/SimdSse41%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -march=corei7 -c -o $@ $< 

SIMD_SSE42_SRCS = $(shell ls $(SIMD_SRC_DIR)/SimdSse42*.cpp)
SIMD_SSE42_OBJS = $(patsubst $(SIMD_SRC_DIR)/%.cpp, $(SIMD_OBJ_DIR)/%.o, $(SIMD_SSE42_SRCS))
SIMD_SSE42_LIB=$(BIN_DIR)/libSimdSse42.a

$(SIMD_SSE42_LIB): $(SIMD_SSE42_OBJS)
	$(DIR_GUARD) 
	$(LNK) -shared -static $(SIMD_SSE42_OBJS) -o $@

$(SIMD_OBJ_DIR)/SimdSse42%.o: $(SIMD_SRC_DIR)/SimdSse42%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -march=corei7 -c -o $@ $< 

SIMD_AVX2_SRCS = $(shell ls $(SIMD_SRC_DIR)/SimdAvx2*.cpp)
SIMD_AVX2_OBJS = $(patsubst $(SIMD_SRC_DIR)/%.cpp, $(SIMD_OBJ_DIR)/%.o, $(SIMD_AVX2_SRCS))
SIMD_AVX2_LIB=$(BIN_DIR)/libSimdAvx2.a

$(SIMD_AVX2_LIB): $(SIMD_AVX2_OBJS)
	$(DIR_GUARD) 
	$(LNK) -shared -static $(SIMD_AVX2_OBJS) -o $@

$(SIMD_OBJ_DIR)/SimdAvx2%.o: $(SIMD_SRC_DIR)/SimdAvx2%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -march=core-avx2 -c -o $@ $< 

SIMD_SRCS = $(SIMD_SRC_DIR)/SimdLib.cpp
SIMD_OBJS = $(patsubst $(SIMD_SRC_DIR)/%.cpp, $(SIMD_OBJ_DIR)/%.o, $(SIMD_SRCS))
SIMD_LIB=$(BIN_DIR)/libSimd.a

$(SIMD_LIB): $(SIMD_OBJS)
	$(DIR_GUARD) 
	$(LNK) -shared -static $(SIMD_OBJS) -o $@

$(SIMD_OBJ_DIR)/%.o: $(SIMD_SRC_DIR)/%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -march=core-avx2 -c -o $@ $< 

TEST_SRCS = $(shell ls $(TEST_SRC_DIR)/*.cpp)
TEST_DEPS = $(shell ls $(TEST_SRC_DIR)/*.h)
TEST_OBJS = $(patsubst $(TEST_SRC_DIR)/%.cpp, $(TEST_OBJ_DIR)/%.o, $(TEST_SRCS))
TEST_EXE=$(BIN_DIR)/Test.exe

$(TEST_EXE): $(TEST_OBJS)
	$(DIR_GUARD) 
	$(LNK) $(TEST_OBJS) $(SIMD_LIB) $(SIMD_BASE_LIB) $(SIMD_SSE2_LIB) $(SIMD_SSSE3_LIB)  $(SIMD_SSE41_LIB)   $(SIMD_SSE42_LIB) $(SIMD_AVX2_LIB) -o $@

$(TEST_OBJ_DIR)/%.o: $(TEST_SRC_DIR)/%.cpp $(TEST_DEPS) $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -march=core-avx2 -std=c++11 -c -o $@ $< 

all: $(SIMD_BASE_LIB) $(SIMD_SSE2_LIB) $(SIMD_SSSE3_LIB) $(SIMD_SSE41_LIB) $(SIMD_SSE42_LIB) $(SIMD_AVX2_LIB) $(SIMD_LIB) $(TEST_EXE) 

clean:
	rm -f -r $(TEST_OBJ_DIR)/* $(SIMD_OBJ_DIR)/* $(BIN_DIR)/*.a $(TEST_EXE)

remake: clean all

