namespace DeepSharp
{
    public class ScalarTensor : Tensor
    {
        /// <summary>
        /// 勾配情報を初期化して生成
        /// </summary>
        /// <param name="value"></param>
        /// <param name="isRequiresGrad"></param>
        /// <param name="name"></param>
        public ScalarTensor(float value, bool isRequiresGrad = false, string name = "NoNameScalarTensor") : base(new float[]{ value }, new int[1]{ 1 }, isRequiresGrad, name) { }
        /// <summary>
        /// 勾配情報を引き継いで生成
        /// </summary>
        /// <param name="t"></param>
        /// <param name="name"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public ScalarTensor(Tensor t, string name = "NoNameScalarTensor") : base(t.Data, t.Shape, t.GradInfo, t.IsRequiresGrad, name)
        {
            if ( !(t.Shape.Length == 1 && t.Data.Length == 1) ) throw new InvalidOperationException("ScalarTensor requires shape length == 0");
        }

        public float Item => Data[0];
    }

    /// <summary>
    /// Tensor1D: shape [C]
    /// </summary>
    public class Tensor1D : Tensor
    {
        /// <summary>
        /// 勾配情報を初期化して生成
        /// </summary>
        /// <param name="data"></param>
        /// <param name="isRequiresGrad"></param>
        /// <param name="name"></param>
        public Tensor1D(float[] data, bool isRequiresGrad = false, string name = "NoNameTensor1D") : base(data, new int[] { data.Length }, isRequiresGrad, name) { }
        /// <summary>
        /// 勾配情報を引き継いで生成
        /// </summary>
        /// <param name="t"></param>
        /// <param name="name"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public Tensor1D(Tensor t, string name = "NoNameTensor1D") : base(t.Data, t.Shape, t.GradInfo, t.IsRequiresGrad, name)
        {
            if (t.Shape.Length != 1) throw new InvalidOperationException("Tensor1D requires shape length == 1");
        }
    }

    /// <summary>
    /// Tensor2D: shape [H, W]
    /// </summary>
    public class Tensor2D : Tensor
    {
        /// <summary>
        /// 勾配情報を初期化して生成
        /// </summary>
        /// <param name="data"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="isRequiresGrad"></param>
        /// <param name="name"></param>
        public Tensor2D(float[] data, int h, int w, bool isRequiresGrad = false, string name = "NoNameTensor2D") : base(data, new int[] { h, w }, isRequiresGrad, name) { }
        /// <summary>
        /// 勾配情報を引き継いで生成
        /// </summary>
        /// <param name="t"></param>
        /// <param name="name"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public Tensor2D(Tensor t, string name = "NoNameTensor2D") : base(t.Data, t.Shape, t.GradInfo, t.IsRequiresGrad, name)
        {
            if (t.Shape.Length != 2) throw new InvalidOperationException("Tensor2D requires shape length == 2");
        }
    }

    /// <summary>
    /// Tensor3D: shape [C, H, W]
    /// </summary>
    public class Tensor3D : Tensor
    {
        /// <summary>
        /// 勾配情報を初期化して生成
        /// </summary>
        /// <param name="data"></param>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="isRequiresGrad"></param>
        /// <param name="name"></param>
        public Tensor3D(float[] data, int c, int h, int w, bool isRequiresGrad = false, string name = "NoNameTensor3D") : base(data, new int[] { c, h, w }, isRequiresGrad, name) { }
        /// <summary>
        /// 勾配情報を引き継いで生成
        /// </summary>
        /// <param name="t"></param>
        /// <param name="name"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public Tensor3D(Tensor t, string name = "NoNameTensor3D") : base(t.Data, t.Shape, t.GradInfo, t.IsRequiresGrad, name)
        {
            if (t.Shape.Length != 3) throw new InvalidOperationException("Tensor3D requires shape length == 3");
        }
    }
}