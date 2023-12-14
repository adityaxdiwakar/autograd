#include <iostream>
#include "val.hpp"

namespace autograd {
    // let doubles be used implicitly
    Value Value::operator+(const double& rhs) const {
        return *this + Value(rhs);
    }

    // allow for change of order val + 12.34 or 12.34 + val
    Value operator+(const double& lhs, const Value& rhs) {
        return rhs + lhs;
    }

    Value Value::operator+(const Value& rhs) const {
        Value other(data + rhs.data);
        return other;
    }
}

int main() {
    autograd::Value a = 3.;
    autograd::Value b = 4.;
    autograd::Value c = a + b;

    std::cout << c.data << std::endl;
}
