using DeepSharp.Layer;

namespace DeepSharp.Model
{
    public class MLP : Model
    {
        public Linear l1, l2;
        public MLP()
        {
            l1 = new Linear(784, 128); // 28x28 → 128
            l2 = new Linear(128, 10);  // 128 → 10
        }

        public BatchTensor1D Forward(BatchTensor1D x)
        {
            var h = Func.ReLU(l1.Forward(x));
            var o = l2.Forward(h);
            return o;
        }

        public Tensor1D Forward(Tensor1D x)
        {
            var h = Func.ReLU(l1.Forward(x));
            var o = l2.Forward(h);
            return o;
        }

        public List<Tensor> GetParameters()
        {
            return new List<Tensor> { l1.weight, l1.bias, l2.weight, l2.bias };
        }
    }
}
