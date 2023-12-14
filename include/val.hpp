#include <functional>
#include <set>

namespace autograd {
    /*
     * Ops contains operations for various possible operations that can occur
     * on the value class. Note, subtraction is supposed using ADD and NEG.
     * Similarly, division is done by using MUL and POW with -1.
     */
    enum class Ops {
        ADD,
        MUL,
        EXP,
        NEG,
        POW,
        RELU,
        TANH
    };

    /**
     * Value is a custom data type that maintains computation graphs in order
     * to iteratively apply chain rule during back prop.
     */
    struct Value {
        Value(double data) : data(data) {}
        Value operator+(Value& rhs);
        Value operator+(double& rhs);

        Value operator*(Value& rhs);
        Value operator*(double& rhs);

        double data = 0;
        double grad = 0; // needs default
        std::function<void()> backward;
        std::set<Value*> parents;
        Ops res_op;
    };

    Value operator+(double& lhs, Value& rhs);
    Value operator*(double& lhs, Value& rhs);
};
