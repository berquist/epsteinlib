# SPDX-FileCopyrightText: 2025 Eric Berquist <eric.berquist@gmail.com>
#
# SPDX-License-Identifier: CC0-1.0

set(TGT epstein)
add_library(${TGT}
    crandall.c
    epsteinZeta.c
    gamma.c
    tools.c
    zeta.c
)
if(UNIX)
    target_link_libraries(${TGT} m)
endif(UNIX)
target_include_directories(${TGT} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>)
install(TARGETS ${TGT})
unset(TGT)

add_subdirectory(tests)
