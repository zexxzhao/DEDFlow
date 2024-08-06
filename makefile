
include config/config.mk

NAME=main
LIBTARGET=libcdam.a
TARGET=
TARGET:=$(TARGET) $(PREFIX)/lib/$(LIBTARGET)
TARGET:=$(TARGET) $(PREFIX)/bin/$(NAME).exe
TARGET:=$(TARGET) $(PREFIX)/bin/mesh_convert.py
TARGET:=$(TARGET) $(PREFIX)/bin/sol2vtk.py

SRC=$(wildcard src/*.c)
# drop src/$NAME.c
SRC:=$(filter-out src/$(NAME).c,$(SRC))
OBJS=$(patsubst src/%.c,obj/%.o,$(SRC))
DEPS=$(patsubst src/%.c,obj/%.d,$(SRC))

CU_SRC=$(wildcard src/*.cu)
CU_OBJS=$(patsubst src/%.cu,obj/%.o,$(CU_SRC))
CU_DEPS=$(patsubst src/%.cu,obj/%.d,$(CU_SRC))

all: $(TARGET)


info:
	@echo "PREFIX: $(PREFIX)"
	@echo "TARGET: $(TARGET)"
	@echo "CC: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "OBJS: $(OBJS)"
	@echo "DEPS: $(DEPS)"
	@echo "SRC: $(SRC)"
	@echo "INC: $(INC)"
	@echo "LIB: $(LIB)"
	@echo "NVCC: $(NVCC)"
	@echo "NVCCFLAGS: $(NVCCFLAGS)"
	@echo "CU_ARCH: $(CU_ARCH)"
	@echo "CU_SRC: $(CU_SRC)"
	@echo "CU_INC: $(CU_INC)"
	@echo "CU_LIB: $(CU_LIB)"
	@echo "CU_OBJS: $(CU_OBJS)"
	@echo "CU_DEPS: $(CU_DEPS)"
	@echo "LINKER: $(LINKER)"


$(PREFIX)/bin/$(NAME).exe: obj/$(NAME).o $(PREFIX)/lib/$(LIBTARGET) | $(PREFIX)/bin
	$(LINKER) $(CFLAGS) $<  $(LDFLAGS) $(LIB) $(CU_LIB) -L$(PREFIX)/lib -lcdam -o $@

$(PREFIX)/bin/%.py: tools/%.py | $(PREFIX)/bin
	cp $< $@

$(PREFIX)/lib/$(LIBTARGET): $(OBJS) $(CU_OBJS) | $(PREFIX)/lib
	$(LINKER) -shared $(CFLAGS) $(CU_OBJS) $(OBJS) $(LIB) $(LDFLAGS) $(CU_LIB) -o $@

obj:
	mkdir -p obj

$(PREFIX)/bin $(PREFIX)/lib $(PREFIX)/include:
	mkdir -p $@

obj/%.o: src/%.c | obj
	$(CC) $(CFLAGS) $(INC) -MMD -c $< -o $@

obj/%.o: src/%.cu | obj
	$(NVCC) $(NVCCFLAGS) $(CU_INC) -MMD -c $< -o $@


-include $(DEPS) $(CU_DEPS) obj/$(NAME).d

clean:
	rm -rf obj $(PREFIX)/include $(PREFIX)/lib $(PREFIX)/bin

.PHONY: info clean install
