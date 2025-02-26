#include <Eigen/Dense>
#include <armadillo> 
#include <iostream>
#include <iomanip>
#include <cmath>

#include "header_file.hpp"

struct Quadrature {
    int n;
    virtual double next() = 0;
};

template<class T> 
struct ext_trap : Quadrature {
    double a, b, s; // a, b = limits of integration, s = current estimated integral value
    double X_A, X_B, l_A, l_B, alpha, beta; // constants for function
    T &func; // func is a reference the function we're integrating

    ext_trap() {}; // default constructor

    // constructor
    ext_trap(T &function, const double lower, const double upper, const double x_a, const double x_b, const double l_a, const double l_b, const double alph, const double bet):
        func(function), a(lower), b(upper), X_A(x_a), X_B(x_b), l_A(l_a), l_B(l_b), alpha(alph), beta(bet) {
            n=0;
    }

    // next() refines the value of the estimated integral each time it is called
    double next(){
        double x, int_num, sum, width;
        int trap_num, j;
        n++;
        // case is n = 1 (only a trapezoid between the two endpoints)
        if (n == 1) {
            return (s = 0.5*(b-a)*(func(a, X_A, X_B, l_A, l_B, alpha, beta)+func(b, X_A, X_B, l_A, l_B, alpha, beta)));
        } else {
            // trap_num = 2^(n−2)
            for (trap_num = 1, j = 1; j < n-1; j++) {
                trap_num <<= 1;
            }

            int_num = trap_num;
            width = (b-a)/int_num; // width of each interval
            x = a + 0.5*width; // evaluate function at midpoint of each interval

            for (sum = 0.0, j = 0; j < trap_num; j++, x += width) {
                sum += func(x, X_A, X_B, l_A, l_B, alpha, beta);
            }

            s = 0.5*(s + (b-a)*sum/int_num); // refining the estimated integral value based on previous sum
            return s;
        }
    }
};


double gaussian_overlap(double x, double X_A, double X_B, double l_A, double l_B, double alpha, double beta){
    return pow(x-X_A, l_A)*pow(x-X_B, l_B)*exp(-alpha*pow(x-X_A, 2) - beta*pow(x-X_B, 2));
}  // overlap integral function

template<class T> 
double refine(T &func, const double a, const double b, double X_A, double X_B, double l_A, double l_B, double alpha, double beta, const double eps=1.0e-10) {
    const int max_j = 20;
    double s, olds = 0.0;
    ext_trap<T> t(func, a, b, X_A, X_B, l_A, l_B, alpha, beta);

    // if l_A or l_B is odd and X_A = X_B then the integral is 0 due to symmetry
    if ((fmod(l_A + l_B, 2.0) != 0.0) && (X_A == X_B)) {
        return 0.0;
    }

    // continue to compute the estimated integral value until convergence
    for (int j = 0; j < max_j; j++) {
        s = t.next();
        if (j > 5)
            if (abs(s-olds) < eps*abs(olds) || (s == 0.0 && olds == 0.0)) 
                return s;
        olds = s;
    }
    throw("Too many steps in routine refine");
}

int main(int argc, char** argv){
    // path to input txt file containing information
    if (argc != 2) {
        std::cerr << "No file path?" << std::endl;
        return 1;
    }

    std::string txt_path = argv[1];
    std::cout << "Input txt file: " << argv[1] << std::endl;

    // read input txt file
    std::ifstream file(txt_path);
    if (!file) {
        std::cerr << "File not found or cannot be opened." << std::endl;}

    arma::mat A;
    A.load(txt_path);
    A.print("Input Matrix: ");

    double inputX_A = A(0,0);
    double inputX_B = A(1,0);
    double inputl_A = A(0,2);
    double inputl_B = A(1,2);
    double inputalpha = A(0,1);
    double inputbeta = A(1,1);

    // set the range to evaluate the integral
    double x_c = ((inputalpha * inputX_A) + (inputbeta * inputX_B))/(inputalpha + inputbeta);
    double adjusted_sd = (50/sqrt(2.0*(inputalpha + inputbeta))) * (1 + (inputl_A + inputl_B)/10.0); // standard deviation (x50) around the center of the product gaussian adjusted to scale on l_A and l_B
    double lower = x_c - adjusted_sd;
    double upper = x_c + adjusted_sd;
    std::cout << "lower bound: " << lower << std::endl;
    std::cout << "upper bound: " << upper << std::endl;

    double final_integral = refine(gaussian_overlap, lower, upper, inputX_A, inputX_B, inputl_A, inputl_B, inputalpha, inputbeta);

    std::cout << "1d numerical overlap integral between Gaussian functions is: " << std::endl;
    std::cout << std::scientific << std::setprecision(15) << final_integral << std::endl;
}
