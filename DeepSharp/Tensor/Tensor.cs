namespace DeepSharp
{
    public partial class Tensor
    {
        public float[] Data;
        public readonly int[] Shape;

        public GradInfo GradInfo;
        public bool IsRequiresGrad { get; set; }

        public string Name { get; set; } = "";

        public Tensor(float[] data, int[] shape, bool isRequiresGrad = false, string name = "NoNameTensor")
        {
            Data = data;
            Shape = shape;
            IsRequiresGrad = isRequiresGrad;
            GradInfo = new GradInfo();
            Name = name;
        }

        public override string ToString()
        {
            return $"Tensor(shape=[{string.Join(",", Shape)}], IsRequiresGrad={IsRequiresGrad}, Name={Name})";
        }
    }
    public static partial class TensorExtension
    {
        public static void Backward(this Tensor tensor)
        {

            // 出発点の勾配は 1（dL/dL = 1）
            if (tensor.GradInfo.Grad == null)
                tensor.GradInfo.Grad = Tensor.OnesLike(tensor);

            // トポロジカル順に並べる
            var topo = Tensor.TopologicalSort(tensor);

            // 後ろから順にBackward
            foreach (var t in topo)
            {
                //Console.WriteLine($"Backward: {t.Name}, IsRequiresGrad={t.IsRequiresGrad}, Parents={string.Join(", ", t.GradInfo.Parents.Select(p => p.Name))}");

                if (t.GradInfo.BackwardFn != null)
                {
                    if (t.GradInfo.Grad == null)
                        throw new InvalidOperationException($"Tensor {t.Name} has null Grad in Backward.");
                    t.GradInfo.BackwardFn(t.GradInfo.Grad);
                }
            }
        }
    }

    public class GradInfo
    {
        /// <summary>
        /// このテンソル自身に対する損失Lの偏微分  ∂𝐿/∂(このテンソル) を保持する場所。計算前なら null
        /// </summary>
        public Tensor? Grad;
        /// <summary>
        /// このテンソルのGradを前の演算Parentsに伝播させる関数。引数はこのテンソルのGrad。ParentsのGradを計算して加算する。
        /// </summary>
        public Action<Tensor>? BackwardFn;
        /// <summary>
        /// このテンソルの計算に使用された親テンソルのリスト。BackwardFn内でこれらのテンソルのGradを計算。
        /// </summary>
        public List<Tensor> Parents;

        public GradInfo()
        {
            Grad = null;
            BackwardFn = null;
            Parents = new List<Tensor>();
        }

        public GradInfo(Tensor? grad, Action<Tensor>? backwardFn, List<Tensor> parents)
        {
            Grad = grad;
            BackwardFn = backwardFn;
            Parents = parents;
        }
    }
}
