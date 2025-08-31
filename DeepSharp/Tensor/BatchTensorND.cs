namespace DeepSharp
{
    /// <summary>
    /// 先頭行がバッチなTensor
    /// </summary>
    public class BatchTensor : Tensor
    {
        public BatchTensor(float[] data, int[] shape, bool isRequiresGrad = false, string name = "NoNameBatchTensor") : base(data, shape, isRequiresGrad, name) { }

        /// <summary>
        /// バッチが複数のTensorで与えられる場合のコンストラクタ
        /// </summary>
        /// <param name="sampleTensors"></param>
        /// <param name="name"></param>
        public BatchTensor(Tensor[] sampleTensors, string name = "NoNameBatchTensor")
            : base(
                  sampleTensors.SelectMany(t => t.Data).ToArray(),
                  ConcatShape(sampleTensors.Length, sampleTensors[0].Shape),
                  sampleTensors[0].IsRequiresGrad,
                  name: name
                  )
        {
        }

        public int BatchSize => Shape[0];
    }

    /// <summary>
    /// BatchTensor1D: shape [B, C]
    /// </summary>
    public class BatchTensor1D : BatchTensor
    {
        public BatchTensor1D(float[] data, int batchSize, int features, bool isRequiresGrad = false, string name = "NoNameBatchTensor1D")
            : base(data, new int[] { batchSize, features }, isRequiresGrad, name) { }

        /// <summary>
        /// バッチが複数のTensor1Dで与えられる場合のコンストラクタ
        /// </summary>
        /// <param name="sampleTensor1Ds"></param>
        public BatchTensor1D(Tensor1D[] sampleTensor1Ds, string name = "NoNameBatchTensor1D")
            : base(sampleTensor1Ds, name)
        {
        }

        public int Features => Shape[1];
    }

    /// <summary>
    /// BatchTensor3D: shape [B, C, H, W]
    /// </summary>
    public class BatchTensor3D : BatchTensor
    {
        public BatchTensor3D(float[] data, int b, int c, int h, int w, bool isRequiresGrad = false, string name = "NoNameBatchTensor3D") : base(data, new int[] { b, c, h, w }, isRequiresGrad, name) { }

        /// <summary>
        /// バッチが複数のTensor3Dで与えられる場合のコンストラクタ
        /// </summary>
        /// <param name="sampleTensor3Ds"></param>
        public BatchTensor3D(Tensor3D[] sampleTensor3Ds, string name = "NoNameBatchTensor3D")
            : base(sampleTensor3Ds, name)
        {
        }

        public int Channels => Shape[1];
        public int Height => Shape[2];
        public int Width => Shape[3];
    }
}