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
unset(TGT)

add_subdirectory(tests)
