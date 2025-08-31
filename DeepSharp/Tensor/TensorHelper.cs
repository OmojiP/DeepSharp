namespace DeepSharp
{
    public partial class Tensor
    {
        /// <summary>
        /// 同じ形状でゼロ埋めのTensorを生成(勾配情報は初期化)
        /// </summary>
        /// <param name="t"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor ZerosLike(Tensor t, string name = "NoNameTensor") => new Tensor(new float[t.Data.Length], (int[])t.Shape.Clone(), t.IsRequiresGrad, name);
        /// <summary>
        /// 同じ形状で1埋めのTensorを生成(勾配情報は初期化)
        /// </summary>
        /// <param name="t"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor OnesLike(Tensor t, string name = "NoNameTensor") => new Tensor(Enumerable.Repeat(1f, t.Data.Length).ToArray(), (int[])t.Shape.Clone(), t.IsRequiresGrad, name);
        /// <summary>
        /// 同じ形状で指定値埋めのTensorを生成(勾配情報は初期化)
        /// </summary>
        /// <param name="t"></param>
        /// <param name="value"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor FullLike(Tensor t, float value, string name = "NoNameTensor") => new Tensor(Enumerable.Repeat(value, t.Data.Length).ToArray(), (int[])t.Shape.Clone(), t.IsRequiresGrad, name);
    }
}
