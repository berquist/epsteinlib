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
target_link_libraries(${TGT} m)
# TODO hack
target_include_directories(${TGT} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
unset(TGT)

add_subdirectory(tests)
