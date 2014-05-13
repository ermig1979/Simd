TRUNK_DIR=../..
SRC_DIR=$(TRUNK_DIR)/src
SIMD_SRC_DIR=$(SRC_DIR)/Simd
TEST_SRC_DIR=$(SRC_DIR)/Test

CXX=g++ -I$(SRC_DIR) -fPIC 
LNK=g++ 
DIR_GUARD=@mkdir -p $(@D)

ifeq ($(DEBUG),1)
CFG=Debug
CXX+= -g -D_DEBUG
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
SIMD_SRCS = $(shell ls $(SIMD_SRC_DIR)/*.cpp)
SIMD_OBJS = $(patsubst $(SIMD_SRC_DIR)/%.cpp, $(SIMD_OBJ_DIR)/%.o, $(SIMD_SRCS))
SIMD_LIB=$(BIN_DIR)/libSimd.so

$(SIMD_LIB): $(SIMD_OBJS)
	$(DIR_GUARD) 
	$(LNK) -shared $(SIMD_OBJS) -o $@

$(SIMD_OBJ_DIR)/SimdBase%.o: $(SIMD_SRC_DIR)/SimdBase%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -c -o $@ $< 

$(SIMD_OBJ_DIR)/SimdSse2%.o: $(SIMD_SRC_DIR)/SimdSse2%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -msse2 -c -o $@ $< 

$(SIMD_OBJ_DIR)/SimdSsse3%.o: $(SIMD_SRC_DIR)/SimdSsse3%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -mssse3 -c -o $@ $< 

$(SIMD_OBJ_DIR)/SimdSse41%.o: $(SIMD_SRC_DIR)/SimdSse41%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -msse4.1 -c -o $@ $< 

$(SIMD_OBJ_DIR)/SimdSse42%.o: $(SIMD_SRC_DIR)/SimdSse42%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -msse4.2 -c -o $@ $< 

$(SIMD_OBJ_DIR)/SimdAvx2%.o: $(SIMD_SRC_DIR)/SimdAvx2%.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -mavx2 -c -o $@ $< 

$(SIMD_OBJ_DIR)/SimdLib.o: $(SIMD_SRC_DIR)/SimdLib.cpp $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -mavx2 -c -o $@ $< 

TEST_SRCS = $(shell ls $(TEST_SRC_DIR)/*.cpp)
TEST_DEPS = $(shell ls $(TEST_SRC_DIR)/*.h)
TEST_OBJS = $(patsubst $(TEST_SRC_DIR)/%.cpp, $(TEST_OBJ_DIR)/%.o, $(TEST_SRCS))
TEST_EXE=$(BIN_DIR)/Test

$(TEST_EXE): $(TEST_OBJS)
	$(DIR_GUARD) 
	$(LNK) $(TEST_OBJS) -l:$(SIMD_LIB) -o $@

$(TEST_OBJ_DIR)/%.o: $(TEST_SRC_DIR)/%.cpp $(TEST_DEPS) $(SIMD_DEPS)
	$(DIR_GUARD)
	$(CXX) -mavx2 -std=c++11 -c -o $@ $< 

all: getversion $(SIMD_LIB) $(TEST_EXE) 

clean:
	rm -f -r $(TEST_OBJ_DIR)/* $(SIMD_OBJ_DIR)/* $(BIN_DIR)/*.a $(BIN_DIR)/*.so $(TEST_EXE)

remake: clean all

getversion:
	sh ./GetVersion.sh




