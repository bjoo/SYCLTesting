/*
 * kokkos_sycl_test1.cpp
 *
 *  Created on: Feb 26, 2020
 *      Author: bjoo
 */

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"

using namespace Kokkos;


TEST(TestKokkos, CreateUSMHostView)
{
	View<float[1024], Experimental::SYCLHostUSMSpace> a("a");
}


TEST(TestKokkos, CreateDynamicExtent)
{
	View<float*, Experimental::SYCLHostUSMSpace> a("a",1024);
}

struct dumb_functor
{
	View<float*, Experimental::SYCLHostUSMSpace> a_;
	void operator()(const int i) const {
		a_(i) *= static_cast<float>(2);
	}
};

TEST(TestKokkos, ParallelFor)
{
	View<float*, Experimental::SYCLHostUSMSpace> a("a",1024);
	for(int i=0; i < 1024; ++i) {
		a(i)=static_cast<float>(i);
	}

	dumb_functor f={a};

	parallel_for(1024,f);

//	for(int i=0; i < 1024; ++i) {
//		ASSERT_FLOAT_EQ( a(i), static_cast<float>(2*i) );
//	}
}
