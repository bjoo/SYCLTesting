#include "gtest/gtest.h"
#include <CL/sycl.hpp>
using namespace cl::sycl;

TEST(SYCLUSMTests, TestBuffer)
{
	gpu_selector my_gpu;
	queue q(my_gpu);

	buffer<float,1> a(1024);
	{
	  auto host_a = a.template get_access<access::mode::write>();
	  for(int i=0; i < 1024; ++i) {
		host_a[i] = static_cast<float>(i);
	  }
	}

	q.submit([&](handler& cgh) {
		auto dev_a = a.template get_access<access::mode::read_write>(cgh);
		cgh.parallel_for(range<1>{1024},[=](id<1> idx) {
			dev_a[idx[0]] *= static_cast<float>(2);
		});
	});

	{
	  auto host_a = a.template get_access<access::mode::read>();
	  for(int i=0; i < 1024; ++i) {
		ASSERT_FLOAT_EQ( host_a[i], static_cast<float>(2*i));
	  }
	}

}

TEST(SYCLUSMTests, TestUSMMallocHost)
{
	gpu_selector my_gpu;
	queue q(my_gpu);

	context ctxt =q.get_context();

	float* a = static_cast<float*>(malloc_host(1024*sizeof(float),ctxt));
	if( a == nullptr) {
		std::cerr << "malloc host returned nullptr" << std::endl;
		FAIL();
	}
	{

	  for(int i=0; i < 1024; ++i) {
		a[i] = static_cast<float>(i);
	  }
	}


	q.submit([&](handler& cgh) {
		cgh.parallel_for(range<1>{1024},[=](id<1> idx) {
			int i=idx[0];
			a[i] *= static_cast<float>(2);
		});
	});
	q.wait();

	for(int i=0; i < 1024; ++i) {
		ASSERT_FLOAT_EQ( a[i], static_cast<float>(2*i));
	}

	free(a,ctxt);
}

TEST(SYCLUSMTests, TestUSMMallocShared)
{
	gpu_selector my_gpu;
	queue q(my_gpu);

	context ctxt =q.get_context();
	device  dev=q.get_device();
	float* a = static_cast<float*>(malloc_shared(1024*sizeof(float),dev,ctxt));
	if( a == nullptr) {
		std::cerr << "malloc host returned nullptr" << std::endl;
		FAIL();
	}
	{

	  for(int i=0; i < 1024; ++i) {
		a[i] = static_cast<float>(i);
	  }
	}


	q.submit([&](handler& cgh) {
		cgh.parallel_for(range<1>{1024},[=](id<1> idx) {
			int i=idx[0];
			a[i] *= static_cast<float>(2);
		});
	});
	q.wait();

	for(int i=0; i < 1024; ++i) {
		ASSERT_FLOAT_EQ( a[i], static_cast<float>(2*i));
	}

	free(a,ctxt);
}

TEST(SYCLUSMTests, TestUSMMallocDevice)
{
	gpu_selector my_gpu;
	queue q(my_gpu);

	context ctxt =q.get_context();
	device  dev=q.get_device();
	float* a_dev= static_cast<float*>(malloc_device(1024*sizeof(float),dev,ctxt));
	if( a_dev == nullptr) {
		std::cerr << "malloc host returned nullptr" << std::endl;
		FAIL();
	}

	{
	  // Regular host array
	  float a[1024];
	  for(int i=0; i < 1024; ++i) {
		a[i] = static_cast<float>(i);
	  }
	  // Copy to a_dev
	  memcpy((void *)a_dev, (const void *)a,1024*sizeof(float));
	}

	// double a_dev on the device
	q.submit([&](handler& cgh) {
		cgh.parallel_for(range<1>{1024},[=](id<1> idx) {
			int i=idx[0];
			a_dev[i] *= static_cast<float>(2);
		});
	});
	q.wait();

	{
		float a[1024];
		// copy a_dev to a
		memcpy((void *)a, (const void *)a_dev, 1024*sizeof(float));
	    for(int i=0; i < 1024; ++i) {
		  ASSERT_FLOAT_EQ( a[i], static_cast<float>(2*i));
	    }
	}
	free(a_dev,ctxt);
}
