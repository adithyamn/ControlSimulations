#include <casadi/casadi.hpp>
#include <iostream>
#include <vector>

using namespace casadi;
using namespace std;

int main() {
    MX x = MX::sym("x",2); // Two states

    // Expression for ODE right-hand side
    MX z = 1 - pow(x(1),2);
    MX rhs = vertcat(z*x(0)-x(1), x(0));

    MXDict ode;         // ODE declaration
    ode["x"]   = x;     // states
    ode["ode"] = rhs;   // right-hand side

    // Construct a Function that integrates over 4s
    Function F = integrator("F","cvodes",ode,0,4);

    // Start from x=[0;1]
    DMDict res = F(DMDict{{"x0",vector<double>{0,1}}});
    cout << "xf = " << res["xf"] << endl;

    // Sensitivity wrt initial state
    MXDict ress = F(MXDict{{"x0",x}});
    Function S("S",{x},{jacobian(ress["xf"],x)});
    cout << "Jacobian = " << S(DM(vector<double>{0,1})) << endl;

    return 0;
}

