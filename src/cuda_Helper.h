#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
// includes, project
#include <assert.h>
#include <exception.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// includes, timer, string parsing, image helpers
#include <helper_image.h>  // helper functions for image compare, dump, data comparisons
#include <helper_string.h>  // helper functions for string parsing
#include <helper_timer.h>   // helper functions for timers
#include <helper_math.h>

#define CUDA_CHECK(call)\
{\
	const cudaError_t errorc = call;\
	if(errorc!=cudaSuccess)\
	{\
		printf("Error: %s:%d", __FILE__, __LINE__);\
		printf("code: %d, reason: %s\n", errorc, cudaGetErrorString(errorc));\
		exit(1);\
	}\
}\

#define THRUST_CHECK(call)\
try\
	{\
		call\
	}\
catch (thrust::system_error e)\
	{\
		std::cout <<"thrust: "<< e.what() << "\n";\
	}\

#define PI 3.1415926f

