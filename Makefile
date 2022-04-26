# User Test
#------------------------------------

APP        = test
APP_SRCS   = matmul.c
APP_INC    = $(TILER_INC)
APP_CFLAGS = -O3

export GAP_USE_OPENOCD=1
include $(RULES_DIR)/pmsis_rules.mk
