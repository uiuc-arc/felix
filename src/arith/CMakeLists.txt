# CMake helper for building rust libraries as a component of a C/C++ project
# Check out https://github.com/ryankurte/rust-cmake for a full example and updated files
# Copyright 2018 Ryan Kurte

# USAGE
# Install nightly rust (`rustup default nightly) and rust-src (`rustup component add rust-src`)
# and make the following additions to your top level CMakeLists.txt
# 1. Include rust.cmake
#     include(rust.cmake)
# 2. Call the build_rust function to build your cargo project
#     build_rust(NAME LOCATION TARGET)
# 3. Link rust libraries to your target
#     target_link_libraries(${CMAKE_PROJECT_NAME} ${RUST_LIBS})

# Locate cargo instance
find_program(CARGO cargo REQUIRED)

function(build_rust RUST_NAME RUST_SOURCE)
    message(STATUS "[RUST] Building Rust project '${RUST_NAME}' from '${RUST_SOURCE}' into '${CMAKE_BINARY_DIR}/${RUST_NAME}'")
    # Set cargo output dir to cmake binary dir (simplifies cleaning etc.)
    set(ARGS --target-dir ${CMAKE_BINARY_DIR}/${RUST_NAME})
    # Native target builds into /target
    set(_RUST_TARGETDIR ${CMAKE_BINARY_DIR}/${RUST_NAME})
    # Native target requires pthread and dl
    set(RUST_LIBS pthread dl)

    # Add release or debug args
    message(STATUS "[RUST] Detected release build")
    set(ARGS ${ARGS} --release)
    set(_RUST_OUTDIR ${_RUST_TARGETDIR}/release)

    add_custom_command(
        OUTPUT ${_RUST_OUTDIR}/lib${RUST_NAME}.a ${_RUST_OUTDIR}/${RUST_NAME}.h 
        COMMAND ${CMAKE_COMMAND} -E env 
        "RUST_HEADER_DIR=${_RUST_TARGETDIR}"
        "RUST_HEADER_NAME=${RUST_NAME}.bin"
        ${CARGO} build ${ARGS}
        WORKING_DIRECTORY ${RUST_SOURCE}
        DEPENDS ${RUST_SOURCE}/*
    )
    add_custom_target(
        ${RUST_NAME}
        DEPENDS ${_RUST_OUTDIR}/lib${RUST_NAME}.a ${_RUST_OUTDIR}/${RUST_NAME}.h
    )
        
    # Add a target to test the rust library
    set(_TEST_CMD "${RUST_NAME}-test")
    add_custom_target(${_TEST_CMD} 
        COMMAND ${CARGO} test ${ARGS}
        WORKING_DIRECTORY ${RUST_SOURCE}
        DEPENDS ${RUST_SOURCE}/*)
    # Add a target to clean the rust library
    set(_CLEAN_CMD "${RUST_NAME}-clean")
    add_custom_target(${_CLEAN_CMD} 
        COMMAND ${CARGO} clean ${ARGS}
        WORKING_DIRECTORY ${RUST_SOURCE}
    )

    # Include the C header output dir
    include_directories(${CMAKE_BINARY_DIR}/c-header-C99)

    # Define a RUST_LIBS variable for linking
    set(RUST_LIBS ${RUST_LIBS} ${_RUST_OUTDIR}/lib${RUST_NAME}.a PARENT_SCOPE)
    # Define a RUST_DEPS variable for specifying dependencies
    # TODO: maybe this could be automated?
    set(RUST_DEPS ${RUST_DEPS} ${RUST_NAME} PARENT_SCOPE)
endfunction(build_rust)

build_rust(egg_simpl "${CMAKE_CURRENT_SOURCE_DIR}/egg_simpl")
add_dependencies(tvm ${RUST_DEPS})
target_link_libraries(tvm PRIVATE ${RUST_LIBS})
