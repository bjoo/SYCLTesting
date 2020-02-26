
#include "Kokkos_Core.hpp"
#include "gtest/gtest.h"



/* This is a convenience routine to setup the test environment for GTest and its layered test environments */
int main(int argc, char **argv)
{

		  ::testing::InitGoogleTest(&argc, argv);
		  Kokkos::initialize(argc,argv);
		  auto ret_val =  RUN_ALL_TESTS();
		  Kokkos::finalize();
		  return ret_val;

}

