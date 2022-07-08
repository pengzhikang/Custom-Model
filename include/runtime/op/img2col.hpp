#include "runtime/engine.hpp"

namespace OCLEngine{
    class Img2colLayer : public CLFunction{
    public:
        Img2colLayer() = default;
        // 配置函数
        void configure();
        // 重载函数，主要的run函数
        void run() override;
    };
}