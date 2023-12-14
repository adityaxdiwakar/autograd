#include <iostream>
#include "val.hpp"

namespace autograd {
    // let doubles be used implicitly
    Value Value::operator+(double& rhs) {
        Value v_rhs(rhs);
        return (*this) + v_rhs;
    }

    // allow for change of order val + 12.34 or 12.34 + val
    Value operator+(double& lhs, Value& rhs) {
        return rhs + lhs;
    }

    Value Value::operator*(double& rhs) {
        Value v_rhs(rhs);
        return (*this) * v_rhs;
    }

    // allow for change of order val + 12.34 or 12.34 + val
    Value operator*(double& lhs, Value& rhs) {
        return rhs * lhs;
    }

    Value Value::operator+(Value& rhs) {
        Value out(data + rhs.data);

        // to track dependencies
        out.parents.insert(this);
        out.parents.insert(&rhs);
        out.res_op = Ops::ADD;

        out.backward = [&]() {
            grad += out.grad;
            rhs.grad += out.grad;
        };

        return out;
    }

    Value Value::operator*(Value& rhs) {
        Value out(data * rhs.data);

        // to track dependencies
        out.parents.insert(this);
        out.parents.insert(&rhs);
        out.res_op = Ops::MUL;

        out.backward = [&]() {
            grad += rhs.data * out.grad;
            rhs.grad += data * out.grad;
        };

        return out;
    }
}

int main() {
    autograd::Value a = 3.;
    autograd::Value b = 4.;
    autograd::Value c = a * b;

    c.grad = 1;
    c.backward();

    std::cout << c.data << std::endl;
    std::cout << c.grad << std::endl;
    std::cout << "a " << a.grad << std::endl;
    std::cout << "b " << b.grad << std::endl;
}
