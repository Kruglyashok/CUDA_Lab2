#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <device_functions.h>
constexpr auto BLOCK_SIZE = 16;

void gpuMatrMult(float* Ad, float* Bd, float* Cd, int rowsA, int colsA, int colsB, float *A, float *B, float *Ch);
__global__ void gpuMatrMultD(float* Ad, float* Bd, float* Cd, int rowsA, int colsA, int colsB);
__global__ void Muld(float*, float*, int, int, float*);
void Mul(const float* A, const float* B, int hA, int wA, int wB, float* C);