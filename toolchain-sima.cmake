
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
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_SYSROOT ${SYSROOT})
set(CMAKE_FIND_ROOT_PATH ${SYSROOT})
set(CMAKE_LIBRARY_ARCHITECTURE aarch64-linux-gnu)
list(PREPEND CMAKE_PREFIX_PATH
    ${SYSROOT}/usr
    ${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake
)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(_SIMA_LMM_SYSROOT_CXX_INCLUDE_FLAGS "")
if(IS_DIRECTORY "${CMAKE_SYSROOT}/usr/include/c++/12")
    string(APPEND _SIMA_LMM_SYSROOT_CXX_INCLUDE_FLAGS
        " -isystem ${CMAKE_SYSROOT}/usr/include/c++/12")
endif()
if(IS_DIRECTORY "${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu/c++/12")
    string(APPEND _SIMA_LMM_SYSROOT_CXX_INCLUDE_FLAGS
        " -isystem ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu/c++/12")
endif()
if(NOT _SIMA_LMM_SYSROOT_CXX_INCLUDE_FLAGS STREQUAL "")
    set(CMAKE_CXX_FLAGS_INIT
        "${CMAKE_CXX_FLAGS_INIT}${_SIMA_LMM_SYSROOT_CXX_INCLUDE_FLAGS}")
endif()

set(_SIMA_LMM_SYSROOT_LINK_FLAGS
    "-L${CMAKE_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/12"
    "-L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu"
    "-L${CMAKE_SYSROOT}/lib/aarch64-linux-gnu"
    "-Wl,-rpath-link,${CMAKE_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/12"
    "-Wl,-rpath-link,${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu"
    "-Wl,-rpath-link,${CMAKE_SYSROOT}/lib/aarch64-linux-gnu")
string(JOIN " " _SIMA_LMM_SYSROOT_LINK_FLAGS_JOINED
    ${_SIMA_LMM_SYSROOT_LINK_FLAGS})
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "${CMAKE_EXE_LINKER_FLAGS_INIT} ${_SIMA_LMM_SYSROOT_LINK_FLAGS_JOINED}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT
    "${CMAKE_SHARED_LINKER_FLAGS_INIT} ${_SIMA_LMM_SYSROOT_LINK_FLAGS_JOINED}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT
    "${CMAKE_MODULE_LINKER_FLAGS_INIT} ${_SIMA_LMM_SYSROOT_LINK_FLAGS_JOINED}")
