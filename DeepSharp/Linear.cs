namespace DeepSharp.Layer
{
    public class Linear
    {
        public Tensor W, b;

        public Linear(int inFeatures, int outFeatures)
        {
            var rand = new Random();
            W = new Tensor(new float[inFeatures * outFeatures], new int[] { inFeatures, outFeatures }) { RequiresGrad = true };
            b = new Tensor(new float[outFeatures], new int[] { 1, outFeatures }) { RequiresGrad = true }; // 修正

            // Xavier初期化
            var limit = (float)Math.Sqrt(6.0 / (inFeatures + outFeatures));
            for (int i = 0; i < W.Data.Length; i++)
                W.Data[i] = (float)(rand.NextDouble() * 2 - 1) * limit;
        }


        public Tensor Forward(Tensor x)
        {
            var y = Tensor.MatMul(x, W);
            return Tensor.AddWithBroadcast(y, b);
        }
    }
}
