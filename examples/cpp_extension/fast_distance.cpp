#include <iostream>
#include <vector>
#include <cmath>

/**
 * Module: Performance Optimization Example (C++)
 * 
 * This is a simple example showing how to implement core logic in C++ 
 * to speed up computation-intensive tasks like distance calculations.
 * 
 * To compile this on Windows (using MinGW/g++):
 * g++ -O3 -shared -static -o fast_distance.dll fast_distance.cpp
 * 
 * To compile on Linux/macOS:
 * g++ -O3 -shared -fPIC -o fast_distance.so fast_distance.cpp
 */

extern "C" {
    /**
     * Calculates the Euclidean distance between two vectors.
     * Using extern "C" ensures the symbol name is preserved for ctypes.
     * 
     * @param a Pointer to the first float array.
     * @param b Pointer to the second float array.
     * @param n Dimensionality of the vectors.
     * @return The Euclidean distance.
     */
    float euclidean_distance(const float* a, const float* b, int n) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    /**
     * Batch calculation of Euclidean distances between a query and many vectors.
     * This better demonstrates the performance advantage of C++.
     * 
     * @param query Pointer to the query vector (size n).
     * @param dataset Pointer to the dataset of vectors (size m * n).
     * @param m Number of vectors in the dataset.
     * @param n Dimensionality of each vector.
     * @param results Pointer to the output array (size m).
     */
    void batch_distance(const float* query, const float* dataset, int m, int n, float* results) {
        for (int i = 0; i < m; ++i) {
            float sum = 0.0f;
            const float* current_vec = dataset + (i * n);
            for (int j = 0; j < n; ++j) {
                float diff = query[j] - current_vec[j];
                sum += diff * diff;
            }
            results[i] = std::sqrt(sum);
        }
    }
}
