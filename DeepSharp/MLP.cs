using DeepSharp.Layer;

namespace DeepSharp.Model
{
    public class MLP
    {
        Linear l1, l2;
        public MLP()
        {
            l1 = new Linear(784, 128); // 28x28 → 128
            l2 = new Linear(128, 10);  // 128 → 10
        }

        public Tensor Forward(Tensor x)
        {
            var h = Tensor.ReLU(l1.Forward(x));
            var o = l2.Forward(h);
            return o;
        }

        public List<Tensor> Parameters()
        {
            return new List<Tensor> { l1.W, l1.b, l2.W, l2.b };
        }
    }
}
