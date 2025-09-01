namespace DeepSharp
{
    public static partial class Func
    {
        public static Tensor Softmax(Tensor t)
        {
            int batch = t.Shape[0], classes = t.Shape[1];
            var resultData = new float[t.Data.Length];
            for (int i = 0; i < batch; i++)
            {
                float max = float.MinValue;
                for (int j = 0; j < classes; j++)
                    max = Math.Max(max, t.Data[i * classes + j]);

                float sumExp = 0;
                for (int j = 0; j < classes; j++)
                    sumExp += (float)Math.Exp(t.Data[i * classes + j] - max);

                for (int j = 0; j < classes; j++)
                    resultData[i * classes + j] = (float)Math.Exp(t.Data[i * classes + j] - max) / sumExp;
            }
            var result = new Tensor(resultData, (int[])t.Shape.Clone(), isRequiresGrad: t.IsRequiresGrad, name: "Softmax");

            if (t.IsRequiresGrad)
            {
                result.GradInfo.Parents = new List<Tensor> { t };
                result.GradInfo.BackwardFn = (Tensor dLdResult) =>
                {
                    if (dLdResult == null)
                        throw new InvalidOperationException("Gradient output is null in Softmax backward function.");
                    // gradOutput.Grad: [B, C]
                    t.GradInfo.Grad ??= Tensor.ZerosLike(t);
                    for (int i = 0; i < batch; i++)
                    {
                        // compute dot = sum_j (dL/dy_j * y_j)
                        float dot = 0f;
                        for (int j = 0; j < classes; j++)
                            dot += dLdResult.Data[i * classes + j] * result.Data[i * classes + j];

                        for (int j = 0; j < classes; j++)
                        {
                            float dy = dLdResult.Data[i * classes + j];
                            float sz = result.Data[i * classes + j];
                            // dz = sz * (dy - dot)
                            t.GradInfo.Grad.Data[i * classes + j] += sz * (dy - dot);
                        }
                    }
                };
            }
            return result;
        }
    }
}
