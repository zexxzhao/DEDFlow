
include config/config.mk

NAME=main
TARGET=$(DEST)/$(NAME).exe

SRC=$(wildcard src/*.c)
OBJS=$(patsubst src/%.c,obj/%.o,$(SRC))
DEPS=$(patsubst src/%.c,obj/%.d,$(SRC))

CU_SRC=$(wildcard src/*.cu)
CU_OBJS=$(patsubst src/%.cu,obj/%.o,$(CU_SRC))
CU_DEPS=$(patsubst src/%.cu,obj/%.d,$(CU_SRC))

all: $(TARGET)


info:
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
	@echo "CU_SRC: $(CU_SRC)"
	@echo "CU_INC: $(CU_INC)"
	@echo "CU_LIB: $(CU_LIB)"
	@echo "CU_OBJS: $(CU_OBJS)"
	@echo "CU_DEPS: $(CU_DEPS)"


$(DEST)/$(NAME).exe: $(OBJS) $(CU_OBJS) | $(DEST)
	$(CC) $(CFLAGS) $(CU_OBJS) $(OBJS) $(LIB) $(LDFLAGS) $(CU_LIB) -o $@ 

obj:
	mkdir -p obj

$(DEST):
	mkdir -p $(DEST)

obj/%.o: src/%.c | obj
	$(CC) $(CFLAGS) $(INC) -MMD -c $< -o $@

obj/%.o: src/%.cu | obj
	$(NVCC) $(NVCCFLAGS) $(CU_INC) -MMD -c $< -o $@


-include $(DEPS) $(CU_DEPS)

clean:
	rm -rf obj $(DEST) $(TARGET)

.PHONY: info clean
