#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <cuComplex.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

// custom kernels
#include "file_io.cuh"
#include "geometry_kernels.cuh"
#include "facet_funcs.cuh"
#include "radar_kernels.cuh"
#include "chirp.cuh"


struct timeval t1, t2;

void startTimer();

void reportTime(const char* msg);

float reportTimeNum();

void reportNewAlloc(int bytes, const char* varName);

void showMemAlloc();

int main(int argc, const char* argv[]);