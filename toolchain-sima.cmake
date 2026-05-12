
# Toolchain file for cross-compiling binaries for SiMa SoC in the Palette docker image.

set(DEFAULT_SYSROOT /opt/toolchain/aarch64/modalix)

if(DEFINED ENV{SYSROOT})
    set(SYSROOT $ENV{SYSROOT})
elseif(EXISTS ${DEFAULT_SYSROOT})
    set(SYSROOT ${DEFAULT_SYSROOT})
else()
    message(
        FATAL_ERROR
        "SYSROOT environment variable not set and the ${DEFAULT_SYSROOT} does not exist"
    )
endif()                

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_PREFIX_PATH ${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake)
set(CMAKE_SYSROOT ${SYSROOT})
set(CMAKE_FIND_ROOT_PATH ${SYSROOT})
set(CMAKE_LIBRARY_ARCHITECTURE aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
