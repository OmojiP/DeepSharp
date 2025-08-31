namespace DeepSharp.Layer
{
    /// <summary>
    /// 全結合層 (Linear Layer)
    /// </summary>
    public class Linear
    {
        public Tensor2D weight;
        public Tensor1D bias;

        public Linear(int inFeatures, int outFeatures)
        {
            var rand = new Random();
            weight = new Tensor2D(new float[inFeatures * outFeatures], inFeatures, outFeatures, isRequiresGrad: true, name: "Linear_weight");
            bias = new Tensor1D(new float[outFeatures], isRequiresGrad: true, name: "Linear_bias");

            // Xavier初期化
            var limit = (float)Math.Sqrt(6.0 / (inFeatures + outFeatures));
            for (int i = 0; i < weight.Data.Length; i++)
                weight.Data[i] = (float)(rand.NextDouble() * 2 - 1) * limit;

            // biasは0で初期化
            for (int i = 0; i < bias.Data.Length; i++)
                bias.Data[i] = 0f;
        }

        public Tensor1D Forward(Tensor1D x)
        {
            var y = Tensor.MatMul(x, weight);
            var result = Tensor.AddWithBroadcast(y, bias);
            return new Tensor1D(result, name: "Linear_Forward_result");
        }

        public BatchTensor1D Forward(BatchTensor1D x)
        {
            var y = Tensor.MatMul(x.ToTensor2D(name: "x_ToTensor2D"), weight);
            var result = Tensor.AddWithBroadcast(y, bias);

            return new BatchTensor1D(result, name: "Linear_Forward_result");
        }
    }
}
