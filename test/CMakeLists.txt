
# Boilerplate from Introduction to modern CMake

# Add the Googletest directory and mark its targets etc as advanced
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")
mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

# Do this to Keep IDE's happy according to introduction to modern cmake
set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)

add_library( testutils test_env.cpp )


        target_link_libraries( testutils kokkos gtest )
        target_include_directories(testutils PUBLIC "${CMAKE_SOURCE_DIR}/include")
        target_include_directories(testutils PUBLIC "${CMAKE_BINARY_DIR}/include")
        target_include_directories(testutils PUBLIC "${Kokkos_SOURCE_DIR}/core/src")
        target_include_directories(testutils PUBLIC "${Kokkos_SOURCE_DIR}/containers/src")
        target_include_directories(testutils PUBLIC "${Kokkos_SOURCE_DIR}/algorithms/src")
        target_include_directories(testutils PUBLIC "${Kokkos_BINARY_DIR}")

# This macro takes the testname and atts an executable from the argumnets
# (ARGN is every argument after the listed one) which allows you to pass a list
# of sources after the testname
macro(package_add_test TESTNAME)
    # Make the executable
    add_executable(${TESTNAME} ${ARGN})

    # Add on the gtest libs
    target_link_libraries(${TESTNAME} testutils)

    # Add the test to CTest
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})

    # I am not 100% sure what tis is for but it is a property
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
  
endmacro()

# Here TESTNAME is test1
# and ARGN is test1.cpp
package_add_test(sycl_usm_tests sycl_usm_tests.cpp)
package_add_test(kokkos_tests kokkos_sycl_test1.cpp)
