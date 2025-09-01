namespace DeepSharp
{
    public partial class Tensor
    {
        public static List<Tensor> TopologicalSort(Tensor root)
        {
            var visited = new HashSet<Tensor>();
            var order = new List<Tensor>();
            void Visit(Tensor t)
            {
                if (!visited.Contains(t))
                {
                    visited.Add(t);
                    foreach (var p in t.GradInfo.Parents) Visit(p);
                    order.Add(t);
                }
            }
            Visit(root);
            order.Reverse();
            return order;
        }

        private static int[] ComputeStrides(int[] shape)
        {
            int n = shape.Length;
            var strides = new int[n];
            int acc = 1;
            for (int i = n - 1; i >= 0; i--)
            {
                strides[i] = acc;
                acc *= shape[i];
            }
            return strides;
        }

        public static Tensor2D Transpose(Tensor2D t)
        {
            int row = t.Shape[0], col = t.Shape[1];
            var resultData = new float[row * col];
            for (int i = 0; i < row; i++)
                for (int j = 0; j < col; j++)
                    resultData[j * row + i] = t.Data[i * col + j];
            // transposeのbackwardはなし？
            return new Tensor2D(resultData, col, row, isRequiresGrad: t.IsRequiresGrad, name: "Transpose");
        }

        public static Tensor Transpose(Tensor t, int[]? perm = null)
        {
            int rank = t.Shape.Length;
            // default: reverse axes
            if (perm == null)
            {
                perm = Enumerable.Range(0, rank).Reverse().ToArray();
            }

            if (perm.Length != rank)
                throw new ArgumentException("Transpose: permutation length must equal tensor rank");

            Tensor result;

            // validate permutation: must be 0..rank-1 each exactly once
            var seen = new bool[rank];
            for (int i = 0; i < rank; i++)
            {
                int p = perm[i];
                if (p < 0 || p >= rank) throw new ArgumentException("Transpose: invalid axis in permutation");
                if (seen[p]) throw new ArgumentException("Transpose: permutation contains duplicate axis");
                seen[p] = true;
            }

            // fast path for 2D swap (same as your original)
            if (rank == 2 && perm[0] == 1 && perm[1] == 0)
            {
                int rows = t.Shape[0], cols = t.Shape[1];
                var resultData = new float[rows * cols];
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        resultData[j * rows + i] = t.Data[i * cols + j];
                result = new Tensor(resultData, new int[] { cols, rows }, isRequiresGrad: t.IsRequiresGrad, name: "Transpose");
                if (t.IsRequiresGrad)
                {
                    result.GradInfo.Parents = new List<Tensor> { t };
                    result.GradInfo.BackwardFn = (Tensor dLdResult) =>
                    {
                        // gradOutput.Grad: same shape as outT
                        if (dLdResult == null) throw new InvalidOperationException("Gradient output is null in Transpose backward function.");
                        t.GradInfo.Grad ??= ZerosLike(t);
                        // gradInput = transpose(gradOutput.Grad)
                        var gradIn = Transpose(dLdResult, new int[] { 1, 0 });
                        AddInto(t.GradInfo.Grad, gradIn);
                    };
                }
                return result;
            }

            // General case:
            int[] outShape = new int[rank];
            for (int i = 0; i < rank; i++) outShape[i] = t.Shape[perm[i]];

            // strides
            var inStrides = ComputeStrides(t.Shape);
            var outStrides = ComputeStrides(outShape);

            int total = 1;
            for (int i = 0; i < outShape.Length; i++) total *= outShape[i];

            var outData = new float[total];

            // iterate linear index in output, map to input index via permuted coords
            for (int outIdx = 0; outIdx < total; outIdx++)
            {
                int rem = outIdx;
                int inIndex = 0;
                for (int dim = 0; dim < rank; dim++)
                {
                    int coord = rem / outStrides[dim];
                    rem %= outStrides[dim];
                    int inDim = perm[dim]; // outCoord for perm[dim]
                    inIndex += coord * inStrides[inDim];
                }
                outData[outIdx] = t.Data[inIndex];
            }

            result = new Tensor(outData, outShape, isRequiresGrad: t.IsRequiresGrad, name: "Transpose");

            if (t.IsRequiresGrad)
            {
                result.GradInfo.Parents = new List<Tensor> { t };
                // compute inverse permutation
                int[] invPerm = new int[rank];
                for (int i = 0; i < rank; i++) invPerm[perm[i]] = i;

                result.GradInfo.BackwardFn = (Tensor dLdResult) =>
                {
                    if (dLdResult == null) throw new InvalidOperationException("Gradient output is null in Transpose backward function.");
                    t.GradInfo.Grad ??= ZerosLike(t);
                    // gradInput = transpose(gradOutput.Grad, inverse permutation)
                    var gradIn = Transpose(dLdResult, invPerm);
                    AddInto(t.GradInfo.Grad, gradIn);
                };
            }

            return result;
        }

        public static void ClipGrad(Model.Model model, float maxNorm)
        {
            double sumsq = 0.0;
            var parameters = model.GetParameters();

            foreach (var p in parameters)
                if (p.GradInfo.Grad != null)
                    for (int i = 0; i < p.GradInfo.Grad.Data.Length; i++)
                        sumsq += p.GradInfo.Grad.Data[i] * p.GradInfo.Grad.Data[i];
            double norm = Math.Sqrt(sumsq);
            if (norm > maxNorm)
            {
                float scale = (float)(maxNorm / (norm + 1e-6));
                foreach (var p in parameters)
                    if (p.GradInfo.Grad != null)
                        for (int i = 0; i < p.GradInfo.Grad.Data.Length; i++)
                            p.GradInfo.Grad.Data[i] *= scale;
            }
        }

        /// <summary>
        /// バッチサイズとサンプルの形状から、バッチ全体の形状を計算して返す
        /// </summary>
        /// <param name="batchSize"></param>
        /// <param name="sampleShape"></param>
        /// <returns></returns>
        public static int[] ConcatShape(int batchSize, int[] sampleShape)
        {
            int[] shape = new int[sampleShape.Length + 1];
            shape[0] = batchSize;
            Array.Copy(sampleShape, 0, shape, 1, sampleShape.Length);
            return shape;
        }
    }
}
