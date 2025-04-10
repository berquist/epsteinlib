# SPDX-FileCopyrightText: 2025 Eric Berquist <eric.berquist@gmail.com>
#
# SPDX-License-Identifier: CC0-1.0

add_library(epstein_test_utils utils.c)

function(add_epstein_test TEST)
    set(TGT "epstein_test_${TEST}")
    add_executable(${TGT} test_${TEST}.c)
    target_link_libraries(${TGT} epstein epstein_test_utils)
    add_test(NAME "${CMAKE_PROJECT_NAME}/${TEST}" COMMAND ${TGT}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    unset(TGT)
endfunction(add_epstein_test)

add_epstein_test(crandall)
add_epstein_test(epsteinZeta)
