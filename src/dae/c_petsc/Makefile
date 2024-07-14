# Default target name (you can override this by specifying TARGET on the command line)
TARGET ?= adjoint

# Compiler
CC = gcc

# Compiler flags
# CFLAGS = -Wall -Wextra -O2
CFLAGS = 

# Libraries to link
LIBS = -lm -lmpi -lpetsc

# Source files (you can override this by specifying SRCS on the command line)
# SRCS ?= adjoint.c
SRCS ?= adjoint.c

# Object files
OBJS = $(SRCS:.c=.o)

# Default rule
all: $(TARGET)

# Rule to build the binary
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Rule to build object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)
