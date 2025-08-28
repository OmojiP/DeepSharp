namespace DeepSharp
{
    public class Tensor
    {
        public float[] Data;
        public int[] Shape;
        public Tensor? Grad;

        public Action<Tensor>? BackwardFn;
        public List<Tensor> Parents;
        public bool RequiresGrad { get; set; }

        public Tensor(float[] data, int[] shape, bool requiresGrad = false)
        {
            Data = data;
            Shape = shape;
            RequiresGrad = requiresGrad;
            Parents = new List<Tensor>();
        }

        // --- ユーティリティ: in-place 加算（勾配を安全に加える） ---
        public static void AddInto(Tensor dest, Tensor src)
        {
            if (dest.Data.Length != src.Data.Length)
                throw new ArgumentException("AddInto: shape mismatch");
            for (int i = 0; i < dest.Data.Length; i++)
                dest.Data[i] += src.Data[i];
        }

        // operator+（完全一致 or bias broadcast [1,C] をサポート）
        public static Tensor operator +(Tensor a, Tensor b)
        {
            if (a.Data.Length == b.Data.Length)
            {
                var res = new Tensor(AddArrays(a.Data, b.Data), (int[])a.Shape.Clone(), a.RequiresGrad || b.RequiresGrad);
                if (a.RequiresGrad || b.RequiresGrad)
                {
                    res.Parents = new List<Tensor> { a, b };
                    res.BackwardFn = (Tensor gradOutput) =>
                    {
                        if (a.RequiresGrad)
                        {
                            a.Grad ??= ZerosLike(a);
                            AddInto(a.Grad, gradOutput.Grad!);
                        }
                        if (b.RequiresGrad)
                        {
                            b.Grad ??= ZerosLike(b);
                            AddInto(b.Grad, gradOutput.Grad!);
                        }
                    };
                }
                return res;
            }

            // b:[1,C], a:[B,C] のブロードキャストのみサポート
            if (b.Shape.Length == 2 && b.Shape[0] == 1 && a.Shape.Length == 2 && a.Shape[1] == b.Shape[1])
            {
                int B = a.Shape[0], C = a.Shape[1];
                var resData = new float[a.Data.Length];
                for (int i = 0; i < B; i++)
                    for (int j = 0; j < C; j++)
                        resData[i * C + j] = a.Data[i * C + j] + b.Data[j];

                var res = new Tensor(resData, (int[])a.Shape.Clone(), a.RequiresGrad || b.RequiresGrad);
                if (a.RequiresGrad || b.RequiresGrad)
                {
                    res.Parents = new List<Tensor> { a, b };
                    res.BackwardFn = (Tensor gradOutput) =>
                    {
                        if (a.RequiresGrad)
                        {
                            a.Grad ??= ZerosLike(a);
                            AddInto(a.Grad, gradOutput.Grad!);
                        }
                        if (b.RequiresGrad)
                        {
                            b.Grad ??= ZerosLike(b); // [1,C]
                                                     // batch方向に集約して加算
                            var tmp = new float[C];
                            for (int i = 0; i < B; i++)
                                for (int j = 0; j < C; j++)
                                    tmp[j] += gradOutput.Grad!.Data[i * C + j];
                            for (int j = 0; j < C; j++)
                                b.Grad.Data[j] += tmp[j];
                        }
                    };
                }
                return res;
            }

            throw new InvalidOperationException("operator + : unsupported shapes");
        }

        // MatMul の backward：in-place AddInto を使用する
        public static Tensor MatMul(Tensor a, Tensor b)
        {
            float[] resultData = MultiplyMatrix(a.Data, a.Shape, b.Data, b.Shape);
            var result = new Tensor(resultData, new int[] { a.Shape[0], b.Shape[1] }, a.RequiresGrad || b.RequiresGrad);

            if (a.RequiresGrad || b.RequiresGrad)
            {
                result.Parents = new List<Tensor> { a, b };
                result.BackwardFn = (Tensor gradOutput) =>
                {
                    // gradOutput.Grad は [B, P]
                    if (a.RequiresGrad)
                    {
                        var gradA = MatMul(gradOutput.Grad!, Transpose(b)); // 新しい Tensor を返す
                        a.Grad ??= ZerosLike(a);
                        AddInto(a.Grad, gradA); // in-place に加算
                    }
                    if (b.RequiresGrad)
                    {
                        var gradB = MatMul(Transpose(a), gradOutput.Grad!);
                        b.Grad ??= ZerosLike(b);
                        AddInto(b.Grad, gradB);
                    }
                };
            }
            return result;
        }


        // --- Zeros/Ones helper ---
        public static Tensor ZerosLike(Tensor t) => new Tensor(new float[t.Data.Length], (int[])t.Shape.Clone(), t.RequiresGrad);
        public static Tensor OnesLike(Tensor t) => new Tensor(Enumerable.Repeat(1f, t.Data.Length).ToArray(), (int[])t.Shape.Clone(), t.RequiresGrad);

        public static float[] AddArrays(float[] a, float[] b)
        {
            var res = new float[a.Length];
            for (int i = 0; i < a.Length; i++) res[i] = a[i] + b[i];
            return res;
        }

        public static float[] MultiplyMatrix(float[] a, int[] ashape, float[] b, int[] bshape)
        {
            int m = ashape[0], n = ashape[1], p = bshape[1];
            var res = new float[m * p];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += a[i * n + k] * b[k * p + j];
                    }
                    res[i * p + j] = sum;
                }
            }
            return res;
        }

        public static Tensor Transpose(Tensor t)
        {
            int rows = t.Shape[0], cols = t.Shape[1];
            var res = new float[rows * cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    res[j * rows + i] = t.Data[i * cols + j];
            return new Tensor(res, new int[] { cols, rows }, t.RequiresGrad);
        }

        // --- ReLU (順/逆伝播をつなぐ) ---
        public static Tensor ReLU(Tensor t)
        {
            var resData = new float[t.Data.Length];
            for (int i = 0; i < resData.Length; i++) resData[i] = Math.Max(0, t.Data[i]);
            var res = new Tensor(resData, (int[])t.Shape.Clone(), t.RequiresGrad);

            if (t.RequiresGrad)
            {
                res.Parents = new List<Tensor> { t };
                res.BackwardFn = (Tensor gradOutput) =>
                {
                    t.Grad ??= ZerosLike(t);
                    for (int i = 0; i < resData.Length; i++)
                    {
                        float gradVal = gradOutput.Grad!.Data[i];
                        t.Grad.Data[i] += (t.Data[i] > 0 ? gradVal : 0f);
                    }
                };
            }
            return res;
        }

        // --- Softmax (順/逆伝播をつなぐ、batch x classes) ---
        public static Tensor Softmax(Tensor t)
        {
            int batch = t.Shape[0], classes = t.Shape[1];
            var resData = new float[t.Data.Length];
            for (int i = 0; i < batch; i++)
            {
                float max = float.MinValue;
                for (int j = 0; j < classes; j++)
                    max = Math.Max(max, t.Data[i * classes + j]);

                float sumExp = 0;
                for (int j = 0; j < classes; j++)
                    sumExp += (float)Math.Exp(t.Data[i * classes + j] - max);

                for (int j = 0; j < classes; j++)
                    resData[i * classes + j] = (float)Math.Exp(t.Data[i * classes + j] - max) / sumExp;
            }
            var res = new Tensor(resData, (int[])t.Shape.Clone(), t.RequiresGrad);

            if (t.RequiresGrad)
            {
                res.Parents = new List<Tensor> { t };
                res.BackwardFn = (Tensor gradOutput) =>
                {
                    // gradOutput.Grad: [B, C]
                    t.Grad ??= ZerosLike(t);
                    for (int i = 0; i < batch; i++)
                    {
                        // compute dot = sum_j (dL/dy_j * y_j)
                        float dot = 0f;
                        for (int j = 0; j < classes; j++)
                            dot += gradOutput.Grad!.Data[i * classes + j] * res.Data[i * classes + j];

                        for (int j = 0; j < classes; j++)
                        {
                            float dy = gradOutput.Grad!.Data[i * classes + j];
                            float sz = res.Data[i * classes + j];
                            // dz = sz * (dy - dot)
                            t.Grad.Data[i * classes + j] += sz * (dy - dot);
                        }
                    }
                };
            }
            return res;
        }

        // --- TopologicalSort はそのまま ---
        public static List<Tensor> TopologicalSort(Tensor root)
        {
            var visited = new HashSet<Tensor>();
            var order = new List<Tensor>();
            void Visit(Tensor t)
            {
                if (!visited.Contains(t))
                {
                    visited.Add(t);
                    foreach (var p in t.Parents) Visit(p);
                    order.Add(t);
                }
            }
            Visit(root);
            order.Reverse();
            return order;
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

            var resData = new float[a.Data.Length];
            for (int i = 0; i < B; i++)
                for (int j = 0; j < C; j++)
                    resData[i * C + j] = a.Data[i * C + j] + b.Data[j]; // works for both b shapes

            var res = new Tensor(resData, (int[])a.Shape.Clone(), a.RequiresGrad || b.RequiresGrad);

            if (a.RequiresGrad || b.RequiresGrad)
            {
                res.Parents = new List<Tensor> { a, b };
                res.BackwardFn = (Tensor gradOutput) =>
                {
                    // gradOutput.Grad : [B, C]
                    if (a.RequiresGrad)
                    {
                        a.Grad ??= ZerosLike(a);
                        AddInto(a.Grad, gradOutput.Grad!); // same shape
                    }
                    if (b.RequiresGrad)
                    {
                        // b may be [C] or [1,C] -> accumulate over batch
                        b.Grad ??= ZerosLike(b);
                        var tmp = new float[C];
                        for (int i = 0; i < B; i++)
                            for (int j = 0; j < C; j++)
                                tmp[j] += gradOutput.Grad!.Data[i * C + j];

                        // add tmp into b.Grad.Data (if b is 1D or 1xC, Data layout is same)
                        for (int j = 0; j < C; j++)
                            b.Grad.Data[j] += tmp[j];
                    }
                };
            }

            return res;
        }

        public static void ClipGrad(List<Tensor> parameters, float maxNorm)
        {
            double sumsq = 0.0;
            foreach (var p in parameters)
                if (p.Grad != null)
                    for (int i = 0; i < p.Grad.Data.Length; i++)
                        sumsq += p.Grad.Data[i] * p.Grad.Data[i];
            double norm = Math.Sqrt(sumsq);
            if (norm > maxNorm)
            {
                float scale = (float)(maxNorm / (norm + 1e-6));
                foreach (var p in parameters)
                    if (p.Grad != null)
                        for (int i = 0; i < p.Grad.Data.Length; i++)
                            p.Grad.Data[i] *= scale;
            }
        }
    }

    public static partial class TensorExtension
    {
        public static void Backward(this Tensor tensor)
        {
            // 出発点の勾配は 1（dL/dL = 1）
            if (tensor.Grad == null)
                tensor.Grad = Tensor.OnesLike(tensor);

            // トポロジカル順に並べる
            var topo = Tensor.TopologicalSort(tensor);

            // 後ろから順にBackward
            foreach (var t in topo)
            {
                if (t.BackwardFn != null)
                {
                    t.BackwardFn(t);
                }
            }
        }
    }
}
