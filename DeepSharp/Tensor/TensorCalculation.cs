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
                    result.GradInfo.BackwardFn = (Tensor dLdResult) =>
                    {
                        if (dLdResult == null)
                            throw new InvalidOperationException("Gradient output is null in operator + backward function.");
                        if (a.IsRequiresGrad)
                        {
                            a.GradInfo.Grad ??= ZerosLike(a);
                            AddInto(a.GradInfo.Grad, dLdResult);
                        }
                        if (b.IsRequiresGrad)
                        {
                            b.GradInfo.Grad ??= ZerosLike(b);
                            AddInto(b.GradInfo.Grad, dLdResult);
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
                    result.GradInfo.BackwardFn = (Tensor dLdResult) =>
                    {
                        if (dLdResult == null)
                            throw new InvalidOperationException("Gradient output is null in operator + backward function.");
                        if (a.IsRequiresGrad)
                        {
                            a.GradInfo.Grad ??= ZerosLike(a);
                            AddInto(a.GradInfo.Grad, dLdResult);
                        }
                        if (b.IsRequiresGrad)
                        {
                            b.GradInfo.Grad ??= ZerosLike(b); // [1,C]
                                                     // batch方向に集約して加算
                            var tmp = new float[C];
                            for (int i = 0; i < B; i++)
                                for (int j = 0; j < C; j++)
                                    tmp[j] += dLdResult.Data[i * C + j];
                            for (int j = 0; j < C; j++)
                                b.GradInfo.Grad.Data[j] += tmp[j];
                        }
                    };
                }
                return result;
            }

            throw new InvalidOperationException("operator + : unsupported shapes");
        }

        /// <summary>
        /// destにsrcをin-placeで加算する。
        /// 形状が異なる場合は例外をスローする。
        /// 偏微分の加算などの勾配を計算したくない計算に使用する。
        /// dL/dz = dL/dz * dz/dx + dL/dz * dz/dy のような場合など。
        /// </summary>
        /// <param name="dest"></param>
        /// <param name="src"></param>
        /// <exception cref="ArgumentException"></exception>
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

        /// <summary>
        /// y = a * b,
        /// a: [K], b: [K, N] -> y: [N]
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static Tensor1D MatMul(Tensor1D a, Tensor2D b)
        {
            int K = a.Shape[0];
            int Kb = b.Shape[0];
            int N = b.Shape[1];
            if (K != Kb) throw new InvalidOperationException($"MatMul(Tensor1D,Tensor2D): inner dim mismatch (a.K={K}, b.K={Kb})");

            // forward: y_j = sum_k a_k * b_{k,j} -> length N
            var outData = new float[N];
            for (int j = 0; j < N; j++)
            {
                float sum = 0f;
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
                result.GradInfo.BackwardFn = (Tensor dLdResult) =>
                {
                    if (dLdResult == null) return; // nothing to do

                    if (a.IsRequiresGrad)
                    {
                        a.GradInfo.Grad ??= ZerosLike(a);
                        // dL/da_k = sum_j dL/dy_j * b_{k,j}
                        // a.Grad[k] += sum_j dLdResult[j] * b[k, j]
                        for (int k = 0; k < K; k++)
                        {
                            float s = 0f;
                            int bOff = k * N;
                            for (int j = 0; j < N; j++)
                                s += dLdResult.Data[j] * b.Data[bOff + j];
                            a.GradInfo.Grad.Data[k] += s;
                        }
                    }

                    // dL/db: K x N
                    if (b.IsRequiresGrad)
                    {
                        b.GradInfo.Grad ??= ZerosLike(b);
                        // dL/db_{k,j} = a_k * dL/dy_j
                        // b.Grad[k, j] += a[k] * dLdResult[j]
                        for (int k = 0; k < K; k++)
                        {
                            float ak = a.Data[k];
                            int bOff = k * N;
                            for (int j = 0; j < N; j++)
                            {
                                b.GradInfo.Grad.Data[bOff + j] += ak * dLdResult.Data[j];
                            }
                        }
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// y = a * b, 
        /// a: [M, N], b: [N, P] -> y: [M, P]
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="isRequiresGrad"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static Tensor2D MatMul(Tensor2D a, Tensor2D b, bool isRequiresGrad = true)
        {
            float[] resultData = MultiplyMatrix(a.Data, a.Shape, b.Data, b.Shape);
            var result = new Tensor2D(resultData, a.Shape[0], b.Shape[1], isRequiresGrad: a.IsRequiresGrad || b.IsRequiresGrad, name: "MatMul");

            if (isRequiresGrad && (a.IsRequiresGrad || b.IsRequiresGrad))
            {
                result.GradInfo.Parents = new List<Tensor> { a, b };
                result.GradInfo.BackwardFn = (Tensor dLdResult) =>
                {
                    if (dLdResult == null)
                        throw new InvalidOperationException("Gradient output is null in MatMul backward function.");

                    var dLdResultTensor2D = dLdResult.ToTensor2D();
                    
                    // gradOutput.Grad は [B, P]
                    if (a.IsRequiresGrad)
                    {
                        // dL/da = dL/dy * b^T
                        var gradA = MatMul(dLdResultTensor2D, Transpose(b), isRequiresGrad: false);
                        // dL/da を a.Grad に加算
                        a.GradInfo.Grad ??= ZerosLike(a);
                        AddInto(a.GradInfo.Grad, gradA);
                    }
                    if (b.IsRequiresGrad)
                    {
                        // dL/db = a^T * dL/dy
                        var gradB = MatMul(Transpose(a), dLdResultTensor2D, isRequiresGrad: false);
                        // dL/db を b.Grad に加算
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
                result.GradInfo.BackwardFn = (Tensor dLdResult) =>
                {
                    if (dLdResult == null)
                        throw new InvalidOperationException("Gradient output is null in AddWithBroadcast backward function.");
                    // gradOutput.Grad : [B, C]
                    if (a.IsRequiresGrad)
                    {
                        a.GradInfo.Grad ??= ZerosLike(a);
                        AddInto(a.GradInfo.Grad, dLdResult); // same shape
                    }
                    if (b.IsRequiresGrad)
                    {
                        // b may be [C] or [1,C] -> accumulate over batch
                        b.GradInfo.Grad ??= ZerosLike(b);
                        var tmp = new float[C];
                        for (int i = 0; i < B; i++)
                            for (int j = 0; j < C; j++)
                                tmp[j] += dLdResult.Data[i * C + j];

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
