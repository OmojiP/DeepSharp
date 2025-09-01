namespace DeepSharp
{
    public static partial class Func
    {
        public static Tensor ReLU(Tensor t)
        {
            var resultData = new float[t.Data.Length];
            for (int i = 0; i < resultData.Length; i++) resultData[i] = Math.Max(0, t.Data[i]);
            var result = new Tensor(resultData, (int[])t.Shape.Clone(), isRequiresGrad: t.IsRequiresGrad, name: "ReLU");

            if (t.IsRequiresGrad)
            {
                result.GradInfo.Parents = new List<Tensor> { t };
                result.GradInfo.BackwardFn = (Tensor dLdResult) =>
                {
                    if (dLdResult == null)
                        throw new InvalidOperationException("Gradient output is null in ReLU backward function.");

                    t.GradInfo.Grad ??= Tensor.ZerosLike(t);
                    for (int i = 0; i < resultData.Length; i++)
                    {
                        float gradVal = dLdResult.Data[i];
                        t.GradInfo.Grad.Data[i] += (t.Data[i] > 0 ? gradVal : 0f);
                    }
                };
            }
            return result;
        }

        public static Tensor1D ReLU(Tensor1D t)
        {
            var resultData = new float[t.Data.Length];
            for (int i = 0; i < resultData.Length; i++) resultData[i] = Math.Max(0, t.Data[i]);
            var result = new Tensor1D(resultData, isRequiresGrad: t.IsRequiresGrad, name: "ReLU");

            if (t.IsRequiresGrad)
            {
                result.GradInfo.Parents = new List<Tensor> { t };
                result.GradInfo.BackwardFn = (Tensor dLdResult) =>
                {
                    if (dLdResult == null)
                        throw new InvalidOperationException("Gradient output is null in ReLU backward function.");

                    t.GradInfo.Grad ??= Tensor.ZerosLike(t);
                    for (int i = 0; i < resultData.Length; i++)
                    {
                        float gradVal = dLdResult.Data[i];
                        t.GradInfo.Grad.Data[i] += (t.Data[i] > 0 ? gradVal : 0f);
                    }
                };
            }
            return result;
        }

        public static BatchTensor1D ReLU(BatchTensor1D t)
        {
            var resultData = new float[t.Data.Length];
            for (int i = 0; i < resultData.Length; i++) resultData[i] = Math.Max(0, t.Data[i]);
            var result = new BatchTensor1D(resultData, t.BatchSize, t.Features, isRequiresGrad: t.IsRequiresGrad, name: "ReLU");

            if (t.IsRequiresGrad)
            {
                result.GradInfo.Parents = new List<Tensor> { t };
                result.GradInfo.BackwardFn = (Tensor dLdResult) =>
                {
                    if (dLdResult == null)
                        throw new InvalidOperationException("Gradient output is null in ReLU backward function.");

                    t.GradInfo.Grad ??= Tensor.ZerosLike(t);
                    for (int i = 0; i < resultData.Length; i++)
                    {
                        float gradVal = dLdResult.Data[i];
                        t.GradInfo.Grad.Data[i] += (t.Data[i] > 0 ? gradVal : 0f);
                    }
                };
            }
            return result;
        }
    }
}
