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
    class Value {
        public:
            Value(double data) : data(data) {}
            Value operator+(const Value& rhs) const;
            Value operator+(const double& rhs) const;

        public: // fields
            double data = 0;
        private:
            double grad_ = 0; // needs default
            std::function<void()> backward_;
            std::set<Value> parents_;
    };

    Value operator+(const double& lhs, const Value& rhs);
};
