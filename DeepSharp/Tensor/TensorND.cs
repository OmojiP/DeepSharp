namespace DeepSharp
{
    public class ScalarTensor : Tensor
    {
        public ScalarTensor(float value, bool isRequiresGrad = false, string name = "NoNameScalarTensor") : base(new float[]{ value }, new int[1]{ 1 }, isRequiresGrad, name) { }

        public float Item => Data[0];
    }

    /// <summary>
    /// Tensor1D: shape [C]
    /// </summary>
    public class Tensor1D : Tensor
    {
        public Tensor1D(float[] data, bool isRequiresGrad = false, string name = "NoNameTensor1D") : base(data, new int[] { data.Length }, isRequiresGrad, name) { }
    }

    /// <summary>
    /// Tensor2D: shape [H, W]
    /// </summary>
    public class Tensor2D : Tensor
    {
        public Tensor2D(float[] data, int h, int w, bool isRequiresGrad = false, string name = "NoNameTensor2D") : base(data, new int[] { h, w }, isRequiresGrad, name) { }
    }

    /// <summary>
    /// Tensor3D: shape [C, H, W]
    /// </summary>
    public class Tensor3D : Tensor
    {
        public Tensor3D(float[] data, int c, int h, int w, bool isRequiresGrad = false, string name = "NoNameTensor3D") : base(data, new int[] { c, h, w }, isRequiresGrad, name) { }
    }
}