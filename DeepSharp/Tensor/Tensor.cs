namespace DeepSharp
{
    public partial class Tensor
    {
        public float[] Data;
        public readonly int[] Shape;

        public GradInfo GradInfo;
        public bool IsRequiresGrad { get; set; }

        public string Name { get; set; } = "";

        public Tensor(float[] data, int[] shape, bool isRequiresGrad = false, string name = "")
        {
            Data = data;
            Shape = shape;
            IsRequiresGrad = isRequiresGrad;
            GradInfo = new GradInfo();
            Name = name;
        }

        public Tensor(float[] data, int[] shape, GradInfo gradInfo, bool isRequiresGrad = false, string name = "")
        {
            Data = data;
            Shape = shape;
            IsRequiresGrad = isRequiresGrad;
            GradInfo = gradInfo;
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
                    t.GradInfo.BackwardFn(t);
                }
            }
        }
    }

    public class GradInfo
    {
        public Tensor? Grad;
        public Action<Tensor>? BackwardFn;
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
