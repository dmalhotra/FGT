
# Directory for SCTL includes
SCTL_INCLUDE_DIR ?= ./extern/SCTL/include

CXX=g++ # requires g++-9 or newer / icpc (with gcc compatibility 9 or newer) / clang++ with llvm-10 or newer
CXXFLAGS = -std=c++17 -fopenmp -Wall -Wfloat-conversion # need C++11 and OpenMP

#Optional flags
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS += -O0 -fsanitize=address,leak,undefined,pointer-compare,pointer-subtract,float-divide-by-zero,float-cast-overflow -fno-sanitize-recover=all -fstack-protector # debug build
	CXXFLAGS += -DSCTL_MEMDEBUG # Enable memory checks
else
	CXXFLAGS += -O3 -march=native -DNDEBUG # release build
endif

OS = $(shell uname -s)
ifeq "$(OS)" "Darwin"
	CXXFLAGS += -g -rdynamic -Wl,-no_pie # for stack trace (on Mac)
else
	CXXFLAGS += -gdwarf-4 -g -rdynamic # for stack trace
endif

CXXFLAGS += -DSCTL_PROFILE=5 -DSCTL_VERBOSE # Enable profiling
CXXFLAGS += -DSCTL_SIG_HANDLER # Enable SCTL stack trace

CXXFLAGS += -DSCTL_QUAD_T=__float128 # Enable quadruple precision

#CXXFLAGS += -lmvec -lm -DSCTL_HAVE_LIBMVEC
#CXXFLAGS += -DSCTL_HAVE_SVML

RM = rm -f
MKDIRS = mkdir -p

BINDIR = ./bin
OBJDIR = ./obj
INCDIR = ./include
TESTDIR = ./test

TARGET_BIN = \
       $(BINDIR)/test

.PHONY: all test clean

all : $(TARGET_BIN)

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX) $^ $(CXXFLAGS) $(LDLIBS) -o $@
ifeq "$(OS)" "Darwin"
	/usr/bin/dsymutil $@ -o $@.dSYM
endif

$(OBJDIR)/%.o: $(TESTDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -I$(SCTL_INCLUDE_DIR) -c $^ -o $@

test: $(TARGET_BIN)
	./$(BINDIR)/test

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~ */*/*~
