#include <casadi/casadi.hpp>
#include <iostream>

int main() {
    using namespace casadi;

    MX x = MX::sym("x");
    MX y = x*x + 1;

    Function f = Function("f", {x}, {y});

    for (int i = 1; i <= 10; i++) {
        DM yi = f(DM(i))[0];  // evaluate numerically
        std::cout << "x = " << i << ", y = " << yi << std::endl;
    }

    return 0;
}

