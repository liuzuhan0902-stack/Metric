import ctypes
import numpy as np
import os
import sys
import time

"""
Module: Python-C++ Interoperability Example

This script demonstrates how to call a C++ function from Python using the 
'ctypes' library. This is useful for those who wish to optimize 
bottleneck operations like custom distance metrics or index traversals.

The demo compares C++ performance against NumPy for a batch of distance 
calculations, showing the speedup obtained by moving the loop to C++.

Instructions:
1. Compile the C++ code first (see fast_distance.cpp for commands).
2. Ensure the compiled shared library (.dll or .so) is in the same directory.
3. Run this script: python test_cpp_ext.py
"""

def main():
    # 1. Path to the shared library
    lib_name = "fast_distance.dll" if sys.platform == "win32" else "fast_distance.so"
    lib_path = os.path.join(os.path.dirname(__file__), lib_name)

    if not os.path.exists(lib_path):
        print(f"Error: Compiled library '{lib_name}' not found.")
        print("Please compile 'fast_distance.cpp' first using the instructions inside the file.")
        return

    # 2. Load the library
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.dirname(lib_path))
    
    try:
        lib = ctypes.CDLL(lib_path)
    except Exception as e:
        print(f"Error: Failed to load library at {lib_path}")
        print(f"Details: {e}")
        return

    # 3. Define argument and return types for the C++ functions
    # float euclidean_distance(const float* a, const float* b, int n)
    lib.euclidean_distance.argtypes = [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_int
    ]
    lib.euclidean_distance.restype = ctypes.c_float

    # void batch_distance(const float* query, const float* dataset, int m, int n, float* results)
    lib.batch_distance.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.batch_distance.restype = None

    # 4. Prepare data (Batch calculation demo)
    dim = 128
    num_vectors = 10000
    query = np.random.rand(dim).astype(np.float32)
    dataset = np.random.rand(num_vectors, dim).astype(np.float32)
    results_cpp = np.zeros(num_vectors, dtype=np.float32)

    print(f"=== C++ Extension Performance Demo ===")
    print(f"Dataset size: {num_vectors} vectors, {dim} dimensions each.")

    # 5. C++ Execution
    start_time = time.time()
    ptr_query = query.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptr_dataset = dataset.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptr_results = results_cpp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    lib.batch_distance(ptr_query, ptr_dataset, num_vectors, dim, ptr_results)
    cpp_time = time.time() - start_time

    # 6. NumPy Execution (Baseline)
    start_time = time.time()
    # Note: NumPy is already very fast as it uses optimized C/Fortran kernels
    results_numpy = np.linalg.norm(dataset - query, axis=1)
    numpy_time = time.time() - start_time

    # 7. Pure Python Execution (To show even more significant speedup)
    # This is often where students' bottlenecks will be if they don't use NumPy correctly.
    start_time = time.time()
    results_python = []
    for i in range(min(num_vectors, 100)): # Only do 100 for python to avoid long wait
        dist = 0.0
        for j in range(dim):
            diff = query[j] - dataset[i][j]
            dist += diff * diff
        results_python.append(np.sqrt(dist))
    python_time_estimated = (time.time() - start_time) * (num_vectors / 100)

    # 8. Comparison and Verification
    diff = np.abs(results_cpp - results_numpy).max()
    
    print(f"\n[Performance Comparison]")
    print(f"C++ Batch Time:      {cpp_time:.6f} seconds")
    print(f"NumPy Time:          {numpy_time:.6f} seconds")
    print(f"Python (Estimated):  {python_time_estimated:.6f} seconds")
    
    print(f"\n[Verification]")
    print(f"Max Difference between C++ and NumPy: {diff:.6e}")
    
    if diff < 1e-5:
        print("Verification PASSED!")
    else:
        print("Verification FAILED!")

    print(f"\nSpeedup (C++ vs Estimated Pure Python): {python_time_estimated / cpp_time:.1f}x")
    print(f"Note: NumPy is extremely optimized. C++ is often used when you need")
    print(f"custom logic that cannot be easily vectorized in NumPy.")

if __name__ == "__main__":
    main()
