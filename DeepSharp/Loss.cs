namespace DeepSharp
{
    public static class Loss
    {
        public static ScalarTensor CrossEntropy(Tensor logits, int[] targets)
        {
            int B = logits.Shape[0], C = logits.Shape[1];
            var lossPer = new float[B];
            var softmax = new float[logits.Data.Length];

            for (int i = 0; i < B; i++)
            {
                // max for numeric stability
                float max = float.MinValue;
                for (int j = 0; j < C; j++)
                    max = Math.Max(max, logits.Data[i * C + j]);

                double sumExp = 0.0;
                for (int j = 0; j < C; j++)
                {
                    double e = Math.Exp(logits.Data[i * C + j] - max);
                    softmax[i * C + j] = (float)e;
                    sumExp += e;
                }
                double logSumExp = Math.Log(sumExp) + max;
                int label = targets[i];
                lossPer[i] = (float)(logSumExp - logits.Data[i * C + label]); // -log P_label
                                                                              // normalize softmax to probabilities for gradient
                for (int j = 0; j < C; j++)
                    softmax[i * C + j] /= (float)sumExp;
            }

            var result = new ScalarTensor(lossPer.Average(), isRequiresGrad: true);
            result.GradInfo.Parents = new List<Tensor> { logits };
            result.GradInfo.BackwardFn = (Tensor gradOutput) =>
            {
                float scale = (gradOutput.GradInfo.Grad != null) ? gradOutput.GradInfo.Grad.Data[0] / (float)B : 1f / (float)B;
                logits.GradInfo.Grad ??= Tensor.ZerosLike(logits);
                for (int i = 0; i < B; i++)
                {
                    int lbl = targets[i];
                    for (int j = 0; j < C; j++)
                    {
                        float y = (j == lbl) ? 1f : 0f;
                        logits.GradInfo.Grad.Data[i * C + j] += (softmax[i * C + j] - y) * scale;
                    }
                }
            };
            return result;
        }
    }
}
