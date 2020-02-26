cmake_minimum_required(VERSION 3.13)
project(SYCLTests VERSION 0.1
                  DESCRIPTION "SYCL_TESTS"
                  LANGUAGES CXX)


# Basic Includes
include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)


# Remember to change the file names
configure_file(include/sycl_tests_config.h.cmake.in include/sycl_tests_config.h)

add_subdirectory(extern/kokkos)
#Make tests here
add_subdirectory(test)


# Install the headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})

# Install the generated headers
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/sycl_test_config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)

# Set some variables to export to the config file if not already set
set(SYCLTests_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
message(STATUS "Setting SYCLTests_INCLUDE_DIR: ${SYCLTests_INCLUDE_DIR}")

# Make the config file
configure_package_config_file(SYCLTestsConfig.cmake.in SYCLTestsConfig.cmake INSTALL_DESTINATION lib/cmake/SYCLTests
 PATH_VARS SYCLTests_INCLUDE_DIR)

# Make the version file
write_basic_package_version_file(
    SYCLTestsVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
    )

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SYCLTestsVersion.cmake ${CMAKE_CURRENT_BINARY_DIR}/SYCLTestsConfig.cmake DESTINATION lib/cmake/SYCLTests)


#install(EXPORT SYCLTestsTargets NAMESPACE SYCLTests:: DESTINATION lib/cmake/SYCLTests )

# Export stuff
# This remains a mystery
export(PACKAGE SYCLTests)
