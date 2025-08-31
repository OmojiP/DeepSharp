namespace DeepSharp
{
    public partial class Tensor
    {
        // operator+（完全一致 or bias broadcast [1,C] をサポート）
        public static Tensor operator +(Tensor a, Tensor b)
        {
            if (a.Data.Length == b.Data.Length)
            {
                var result = new Tensor(AddArrays(a.Data, b.Data), (int[])a.Shape.Clone(), isRequiresGrad: a.IsRequiresGrad || b.IsRequiresGrad, name: "Add");
                if (a.IsRequiresGrad || b.IsRequiresGrad)
                {
                    result.GradInfo.Parents = new List<Tensor> { a, b };
                    result.GradInfo.BackwardFn = (Tensor gradOutput) =>
                    {
                        if (gradOutput.GradInfo.Grad == null)
                            throw new InvalidOperationException("Gradient output is null in operator + backward function.");
                        if (a.IsRequiresGrad)
                        {
                            a.GradInfo.Grad ??= ZerosLike(a);
                            AddInto(a.GradInfo.Grad, gradOutput.GradInfo.Grad);
                        }
                        if (b.IsRequiresGrad)
                        {
                            b.GradInfo.Grad ??= ZerosLike(b);
                            AddInto(b.GradInfo.Grad, gradOutput.GradInfo.Grad);
                        }
                    };
                }
                return result;
            }

            // b:[1,C], a:[B,C] のブロードキャストのみサポート
            if (b.Shape.Length == 2 && b.Shape[0] == 1 && a.Shape.Length == 2 && a.Shape[1] == b.Shape[1])
            {
                int B = a.Shape[0], C = a.Shape[1];
                var resultData = new float[a.Data.Length];
                for (int i = 0; i < B; i++)
                    for (int j = 0; j < C; j++)
                        resultData[i * C + j] = a.Data[i * C + j] + b.Data[j];

                var result = new Tensor(resultData, (int[])a.Shape.Clone(), isRequiresGrad: a.IsRequiresGrad || b.IsRequiresGrad, name: "Add");
                if (a.IsRequiresGrad || b.IsRequiresGrad)
                {
                    result.GradInfo.Parents = new List<Tensor> { a, b };
                    result.GradInfo.BackwardFn = (Tensor gradOutput) =>
                    {
                        if (gradOutput.GradInfo.Grad == null)
                            throw new InvalidOperationException("Gradient output is null in operator + backward function.");
                        if (a.IsRequiresGrad)
                        {
                            a.GradInfo.Grad ??= ZerosLike(a);
                            AddInto(a.GradInfo.Grad, gradOutput.GradInfo.Grad);
                        }
                        if (b.IsRequiresGrad)
                        {
                            b.GradInfo.Grad ??= ZerosLike(b); // [1,C]
                                                     // batch方向に集約して加算
                            var tmp = new float[C];
                            for (int i = 0; i < B; i++)
                                for (int j = 0; j < C; j++)
                                    tmp[j] += gradOutput.GradInfo.Grad.Data[i * C + j];
                            for (int j = 0; j < C; j++)
                                b.GradInfo.Grad.Data[j] += tmp[j];
                        }
                    };
                }
                return result;
            }

            throw new InvalidOperationException("operator + : unsupported shapes");
        }

        public static void AddInto(Tensor dest, Tensor src)
        {
            if (dest.Data.Length != src.Data.Length)
                throw new ArgumentException("AddInto: shape mismatch");
            for (int i = 0; i < dest.Data.Length; i++)
                dest.Data[i] += src.Data[i];
        }

        private static float[] AddArrays(float[] a, float[] b)
        {
            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++) result[i] = a[i] + b[i];
            return result;
        }


        public static BatchTensor1D MatMul(BatchTensor1D a, Tensor2D b)
        {
            var result = MatMul(a.ToTensor2D(), b);
            return result.ToBatchTensor1D();
        }

        public static Tensor1D MatMul(Tensor1D a, Tensor2D b)
        {
            int K = a.Shape[0];
            int Kb = b.Shape[0];
            int N = b.Shape[1];
            if (K != Kb) throw new InvalidOperationException($"MatMul(Tensor1D,Tensor2D): inner dim mismatch (a.K={K}, b.K={Kb})");

            // forward: y_j = sum_k a_k * b_{k,j}  -> length N
            var outData = new float[N];
            for (int j = 0; j < N; j++)
            {
                float sum = 0f;
                int baseIndex = j; // we'll access b as b[k*N + j]
                for (int k = 0; k < K; k++)
                {
                    sum += a.Data[k] * b.Data[k * N + j];
                }
                outData[j] = sum;
            }

            var result = new Tensor1D(outData, isRequiresGrad: a.IsRequiresGrad || b.IsRequiresGrad, name: "MatMul");

            if (a.IsRequiresGrad || b.IsRequiresGrad)
            {
                result.GradInfo.Parents = new List<Tensor> { a, b };
                result.GradInfo.BackwardFn = (Tensor gradOutput) =>
                {
                    if (gradOutput.GradInfo.Grad == null) return; // nothing to do

                    var go = gradOutput.GradInfo.Grad; // shape [N]

                    // dL/da: length K
                    if (a.IsRequiresGrad)
                    {
                        a.GradInfo.Grad ??= ZerosLike(a);
                        // a.Grad[k] += sum_j go[j] * b[k, j]
                        for (int k = 0; k < K; k++)
                        {
                            float s = 0f;
                            int bOff = k * N;
                            for (int j = 0; j < N; j++)
                                s += go.Data[j] * b.Data[bOff + j];
                            a.GradInfo.Grad.Data[k] += s;
                        }
                    }

                    // dL/db: K x N
                    if (b.IsRequiresGrad)
                    {
                        b.GradInfo.Grad ??= ZerosLike(b);
                        // b.Grad[k, j] += a[k] * go[j]
                        for (int k = 0; k < K; k++)
                        {
                            float ak = a.Data[k];
                            int bOff = k * N;
                            for (int j = 0; j < N; j++)
                            {
                                b.GradInfo.Grad.Data[bOff + j] += ak * go.Data[j];
                            }
                        }
                    }
                };
            }

            return result;
        }

        public static Tensor2D MatMul(Tensor2D a, Tensor2D b)
        {
            float[] resultData = MultiplyMatrix(a.Data, a.Shape, b.Data, b.Shape);
            var result = new Tensor2D(resultData, a.Shape[0], b.Shape[1], isRequiresGrad: a.IsRequiresGrad || b.IsRequiresGrad, name: "MatMul");

            if (a.IsRequiresGrad || b.IsRequiresGrad)
            {
                result.GradInfo.Parents = new List<Tensor> { a, b };
                result.GradInfo.BackwardFn = (Tensor gradOutput) =>
                {
                    // gradOutput.Grad は [B, P]
                    if (a.IsRequiresGrad)
                    {
                        var gradOutputTensor2D = gradOutput.GradInfo.Grad!.ToTensor2D();
                        
                        var gradA = MatMul(gradOutputTensor2D, Transpose(b)); // 新しい Tensor を返す
                        a.GradInfo.Grad ??= ZerosLike(a);
                        AddInto(a.GradInfo.Grad, gradA); // in-place に加算
                    }
                    if (b.IsRequiresGrad)
                    {
                        var gradOutputTensor2D = gradOutput.GradInfo.Grad!.ToTensor2D();
                                                
                        var gradB = MatMul(Transpose(a), gradOutputTensor2D);
                        b.GradInfo.Grad ??= ZerosLike(b);
                        AddInto(b.GradInfo.Grad, gradB);

                    }
                };
            }
            return result;
        }

        private static float[] MultiplyMatrix(float[] a, int[] ashape, float[] b, int[] bshape)
        {
            int m = ashape[0], n = ashape[1], p = bshape[1];
            var result = new float[m * p];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += a[i * n + k] * b[k * p + j];
                    }
                    result[i * p + j] = sum;
                }
            }
            return result;
        }

        public static Tensor AddWithBroadcast(Tensor a, Tensor b)
        {
            // a: [B, C], b: [1, C] or [C]
            if (!(a.Shape.Length == 2 && a.Shape[1] > 0))
                throw new InvalidOperationException("AddWithBroadcast: a must be 2D");

            int B = a.Shape[0], C = a.Shape[1];

            // accept b shaped [1,C] or [C]
            bool bIs1D = (b.Shape.Length == 1 && b.Shape[0] == C);
            bool bIs1xC = (b.Shape.Length == 2 && b.Shape[0] == 1 && b.Shape[1] == C);
            if (!bIs1D && !bIs1xC)
                throw new InvalidOperationException("AddWithBroadcast: b must be shape [C] or [1,C]");

            var resultData = new float[a.Data.Length];
            for (int i = 0; i < B; i++)
                for (int j = 0; j < C; j++)
                    resultData[i * C + j] = a.Data[i * C + j] + b.Data[j]; // works for both b shapes

            var result = new Tensor(resultData, (int[])a.Shape.Clone(), isRequiresGrad: a.IsRequiresGrad || b.IsRequiresGrad, name: "AddWithBroadcast");

            if (a.IsRequiresGrad || b.IsRequiresGrad)
            {
                result.GradInfo.Parents = new List<Tensor> { a, b };
                result.GradInfo.BackwardFn = (Tensor gradOutput) =>
                {
                    if (gradOutput.GradInfo.Grad == null)
                        throw new InvalidOperationException("Gradient output is null in AddWithBroadcast backward function.");
                    // gradOutput.Grad : [B, C]
                    if (a.IsRequiresGrad)
                    {
                        a.GradInfo.Grad ??= ZerosLike(a);
                        AddInto(a.GradInfo.Grad, gradOutput.GradInfo.Grad); // same shape
                    }
                    if (b.IsRequiresGrad)
                    {
                        // b may be [C] or [1,C] -> accumulate over batch
                        b.GradInfo.Grad ??= ZerosLike(b);
                        var tmp = new float[C];
                        for (int i = 0; i < B; i++)
                            for (int j = 0; j < C; j++)
                                tmp[j] += gradOutput.GradInfo.Grad.Data[i * C + j];

                        // add tmp into b.Grad.Data (if b is 1D or 1xC, Data layout is same)
                        for (int j = 0; j < C; j++)
                            b.GradInfo.Grad.Data[j] += tmp[j];
                    }
                };
            }

            return result;
        }
    }
}
