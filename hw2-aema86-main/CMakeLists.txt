cmake_minimum_required (VERSION 3.10)

project (hw1)

# Compiler requirements
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Linear Algebra Header Libraries
find_package(Armadillo REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find Linear Algebra Runtime Libraries
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Configure Eigen to use BLAS & LAPACKE (if you need, you can remove this. Eigen does not need these strictly.)
add_compile_definitions(EIGEN_USE_BLAS EIGEN_USE_LAPACKE)

# Includes And Links
include_directories(${ARMADILLO_INCLUDE_DIRS})
link_libraries(${ARMADILLO_LIBRARIES})
link_libraries(Eigen3::Eigen) 

add_subdirectory(src)

