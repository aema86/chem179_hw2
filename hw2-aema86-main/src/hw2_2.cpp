#include <Eigen/Dense>
#include <armadillo> 
#include <iostream>
#include <cmath>
#include <numbers>
#include <vector>

#include "header_file.hpp"

// BASIC FUNCTIONS

// double factorial function
double doublefactorial(int n) {
    if (n <= 0) return 1.0;
    return n * doublefactorial(n - 2);
}

// factorial function
double factorial(unsigned int n) {
    if (n < 0) return 1.0;
    if (n == 0 || n == 1) return 1.0;
    return n * factorial(n - 1);
}

// combination function
double combination(int l, int i) {
    return static_cast<double>(factorial(l)) / (factorial(i) * factorial(l - i));
}

// S_ABx (1D overlap integral) COMPONENTS 

// Rp (Xp, Yp, Zp) aka the center of the product vector function
std::vector<double> Rp(const arma::mat& matrix) {
    // identifying alpha and beta coefficients from the matrix
    double alpha = matrix(0,3);
    double beta = matrix(1,3);

    // identifying coordinates from the matrix
    double x_A = matrix(0,0);
    double y_A = matrix(0,1);
    double z_A = matrix(0,2);
    double x_B = matrix(1,0);
    double y_B = matrix(1,1);
    double z_B = matrix(1,2);

    // output the Rp into a vector in the form of: {Xp, Yp, Zp}
    std::vector<double> Rp_vector;
    double Xp = (alpha*x_A + beta*x_B)/(alpha + beta);
    double Yp = (alpha*y_A + beta*y_B)/(alpha + beta);
    double Zp = (alpha*z_A + beta*z_B)/(alpha + beta);
    Rp_vector.push_back(Xp);
    Rp_vector.push_back(Yp);
    Rp_vector.push_back(Zp);

    //std::cout << "Rp values: " << Rp_vector[0] << ", " << Rp_vector[1] << ", " << Rp_vector[2] << std::endl;

    return Rp_vector;
}

// starting coefficients (s_x, s_y, s_z) of sab (1D overlap integral) vector function
std::vector<double> starting(arma::mat matrix) {
    double alpha = matrix(0, 3);
    double beta = matrix(1, 3);

    double x_A = matrix(0, 0);
    double y_A = matrix(0, 1);
    double z_A = matrix(0, 2);
    double x_B = matrix(1, 0);
    double y_B = matrix(1, 1);
    double z_B = matrix(1, 2);

    std::vector<double> coeffs_vector;

    // output the starting coefficients into a vector in the form of: {s_x, s_y, s_z}
    double s_x = std::sqrt(std::numbers::pi / (alpha + beta)) * exp(-(alpha * beta * pow(x_A - x_B, 2)) / (alpha + beta));
    double s_y = std::sqrt(std::numbers::pi / (alpha + beta)) * exp(-(alpha * beta * pow(y_A - y_B, 2)) / (alpha + beta));
    double s_z = std::sqrt(std::numbers::pi / (alpha + beta)) * exp(-(alpha * beta * pow(z_A - z_B, 2)) / (alpha + beta));

    coeffs_vector.push_back(s_x);
    coeffs_vector.push_back(s_y);
    coeffs_vector.push_back(s_z);

    //std::cout << "Starting coefficients: s_x = " << coeffs_vector[0] << ", s_y = " << coeffs_vector[1] << ", s_z = " << coeffs_vector[2] << std::endl;

    return coeffs_vector;
}

// calculating the 3D overlap integral for one set of (la, ma, na) and (lb, mb, nb) aka angular momentum components
double one_overlap(std::vector<double> s_vector, std::vector<double> R_vector, const arma::mat& matrix, std::vector<int> A_lmn, std::vector<int> B_lmn){
    // pulling out all the necessary values from the matrix (exponents, coordinates)
    double alpha = matrix(0, 3);
    double beta = matrix(1, 3);

    double x_A = matrix(0, 0);
    double y_A = matrix(0, 1);
    double z_A = matrix(0, 2);
    double x_B = matrix(1, 0);
    double y_B = matrix(1, 1);
    double z_B = matrix(1, 2);

    //std::cout << "Alpha: " << alpha << ", Beta: " << beta << std::endl;
    //std::cout << "X_A: " << x_A << ", Y_A: " << y_A << ", Z_A: " << z_A << std::endl;
    //std::cout << "X_B: " << x_B << ", Y_B: " << y_B << ", Z_B: " << z_B << std::endl;
    
    // pulling out all the necessary values from the vectors (R_p, starting coefficients, l,m,n values per shell)
    double s_x = s_vector[0];
    double s_y = s_vector[1];
    double s_z = s_vector[2];

    double x_p = R_vector[0];
    double y_p = R_vector[1];
    double z_p = R_vector[2];

    int la = A_lmn[0];
    int ma = A_lmn[1];
    int na = A_lmn[2];

    int lb = B_lmn[0];
    int mb = B_lmn[1];
    int nb = B_lmn[2];

    double sab_x = 0.0, sab_y = 0.0, sab_z = 0.0;
    
    // double summation portion of the calculation of one overlap integral along the x coordinate
    for (int i = 0; i <= la; ++i) {
        for (int j = 0; j <= lb; ++j) {
            if ((i+j) % 2 == 0){
                double term_x = combination(la, i) * combination(lb, j) * (doublefactorial(i+j-1) * pow(x_p-x_A, (la-i)) * pow(x_p-x_B, (lb-j)))/(pow(2.0*(alpha+beta), (i+j)/2.0));
                //std::cout << "term_x: " << term_x << std::endl;
                sab_x += term_x;
                //std::cout << "sab_x term (" << i << "," << j << "): " << term_x << std::endl;  // debugging
            }
        }
    }
    //std::cout << "sab_x: " << sab_x << std::endl;

    // double summation portion of the calculation of one overlap integral along the y coordinate
    for (int i = 0; i <= ma; ++i) {
        for (int j = 0; j <= mb; ++j) {
            if ((i+j) % 2 == 0){
                double term_y = combination(ma, i) * combination(mb, j) * (doublefactorial(i+j-1) * pow(y_p-y_A, (ma-i)) * pow(y_p-y_B, (mb-j)))/(pow(2.0*(alpha+beta), (i+j)/2.0));
                //std::cout << "term_y: " << term_y << std::endl;
                sab_y += term_y;
            }
        }
    }
    //std::cout << "sab_y: " << sab_y << std::endl;

    // double summation portion of the calculation of one overlap integral along the z coordinate
    for (int i = 0; i <= na; ++i) {
        for (int j = 0; j <= nb; ++j) {
            if ((i+j) % 2 == 0){
                double term_z = combination(na, i) * combination(nb, j) * (doublefactorial(i+j-1) * pow(z_p-z_A, (na-i)) * pow(z_p-z_B, (nb-j)))/(pow(2.0*(alpha+beta), (i+j)/2.0));
                //std::cout << "term_z: " << term_z << std::endl;
                sab_z += term_z;
            }
        }
    }
    //std::cout << "sab_z: " << sab_z << std::endl;

    // final calculation of the overlap integral sab
    double sab = sab_x * s_x * sab_y * s_y * sab_z * s_z;
    //std::cout << "overlap: " << sab << std::endl;
    return sab;
}

int main(int argc, char** argv){
    // define the input matrix (from txt file)
    arma::mat A(2, 5);
    A << 0.0 << 0.0 << 0.0 << 1.0 << 0.0 << arma::endr
      << 0.0 << 0.0 << 0.0 << 1.0 << 0.0 << arma::endr;

    // math check - checking if math functions are working properly 
    std::vector<double> math_1 = {doublefactorial(0), doublefactorial(1), doublefactorial(5), doublefactorial(-1)};
    std::vector<double> math_2 = {factorial(0), factorial(1), factorial(4), factorial(5)};
    std::vector<double> math_3 = {combination(1,0), combination(1,1), combination(2, 2), combination(6,3)};

    /* std::cout << "double factorial test" << std::endl;
    for (double num : math_1) {
        std::cout << num << " " << std::endl;
    }

    std::cout << "factorial test" << std::endl;
    for (double num : math_2) {
        std::cout << num << " " << std::endl;
    }

    std::cout << "combination test" << std::endl;
    for (double num : math_3) {
        std::cout << num << " " << std::endl;
    } */

    // identifying l, m and n depending on L from the input matrix

    int A_function_num = 0;
    arma::mat A_shell_funcs;
    // l, m, n for shell A
    // outputs all possible l, m, n arrangements in a matrix with 3 columns and A_function_num rows
    if (A(0, 4) == 2) {
        A_function_num = 6;
        A_shell_funcs.set_size(6, 3);
        A_shell_funcs << 2 << 0 << 0 << arma::endr
                      << 0 << 2 << 0 << arma::endr
                      << 0 << 0 << 2 << arma::endr
                      << 1 << 1 << 0 << arma::endr
                      << 0 << 1 << 1 << arma::endr
                      << 1 << 0 << 1 << arma::endr;
    } else if (A(0, 4) == 1) {
        A_function_num = 3;
        A_shell_funcs.set_size(3, 3);
        A_shell_funcs << 1 << 0 << 0 << arma::endr
                      << 0 << 1 << 0 << arma::endr
                      << 0 << 0 << 1 << arma::endr;
    } else if (A(0, 4) == 0) {
        A_function_num = 1;
        A_shell_funcs.set_size(1, 3);
        A_shell_funcs << 0 << 0 << 0 << arma::endr;
    } else {
        std::cerr << "Shell is too complex! (not s, p or d)" << std::endl;
    }

    int B_function_num = 0;
    arma::mat B_shell_funcs;
    // l, m, n for shell B
    // outputs all possible l, m, n arrangements in a matrix with 3 columns and B_function_num rows
    if (A(1, 4) == 2) {
        B_function_num = 6;
        B_shell_funcs.set_size(6, 3);
        B_shell_funcs << 2 << 0 << 0 << arma::endr
                      << 0 << 2 << 0 << arma::endr
                      << 0 << 0 << 2 << arma::endr
                      << 1 << 1 << 0 << arma::endr
                      << 0 << 1 << 1 << arma::endr
                      << 1 << 0 << 1 << arma::endr;
    } else if (A(1, 4) == 1) {
        B_function_num = 3;
        B_shell_funcs.set_size(3, 3);
        B_shell_funcs << 1 << 0 << 0 << arma::endr
                      << 0 << 1 << 0 << arma::endr
                      << 0 << 0 << 1 << arma::endr;
    } else if (A(1, 4) == 0) {
        B_function_num = 1;
        B_shell_funcs.set_size(1, 3);
        B_shell_funcs << 0 << 0 << 0 << arma::endr;
    } else {
        std::cerr << "Shell is too complex! (not s, p or d)" << std::endl;
    }

    std::cout << "The components of angular momentum (l, m, n) for the overlap matrix column, from top to bottom, are listed along the rows of:\n" << A_shell_funcs << std::endl;
    std::cout << "The components of angular momentum (l, m, n) for the overlap matrix row, from left to right, are listed along the rows of:\n" << B_shell_funcs << std::endl;

    // define vectors for R_p and starting coefficients
    std::vector<double> rp = Rp(A);
    std::vector<double> sc = starting(A);

    double final;
    arma::mat fin(A_shell_funcs.n_rows, B_shell_funcs.n_rows, arma::fill::zeros);
    // iterate and implement one_overlap function to make overlap matrix
    for (int i = 0; i < A_shell_funcs.n_rows; ++i) {
        for (int j = 0; j < B_shell_funcs.n_rows; ++j) {
            //std::cout << "(A shell index, B shell index)" << "(" << i << ", " << j << ")\n";
            std::vector<int> A1 = arma::conv_to<std::vector<int>>::from(A_shell_funcs.row(i));
            /* std::cout << "A1: " << std::endl;
            for (int num : A1) {
                std::cout << num << " ";
            }
            std::cout << std::endl; */
         
            std::vector<int> B1 = arma::conv_to<std::vector<int>>::from(B_shell_funcs.row(j));
            /* std::cout << "B1: " << std::endl;
            for (int num : B1) {
                std::cout << num << " ";
            }
            std::cout << std::endl; */

            final = one_overlap(sc, rp, A, A1, B1);
            fin(i, j) = final;
            //std::cout << final << std::endl;
        }
    }
    fin.print("Overlap integral between Shell 1 and Shell 2: ");
}
