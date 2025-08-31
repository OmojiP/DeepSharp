namespace DeepSharp
{
    /// <summary>
    /// 先頭行がバッチなTensor
    /// </summary>
    public class BatchTensor : Tensor
    {
        /// <summary>
        /// 勾配情報を初期化して生成
        /// </summary>
        /// <param name="data"></param>
        /// <param name="shape"></param>
        /// <param name="isRequiresGrad"></param>
        /// <param name="name"></param>
        public BatchTensor(float[] data, int[] shape, bool isRequiresGrad = false, string name = "NoNameBatchTensor") : base(data, shape, isRequiresGrad, name) { }
        /// <summary>
        /// 勾配情報を引き継いで生成
        /// </summary>
        /// <param name="batchTensor"></param>
        /// <param name="name"></param>
        public BatchTensor(Tensor batchTensor, string name = "NoNameBatchTensor") : base(batchTensor.Data, batchTensor.Shape, batchTensor.GradInfo, batchTensor.IsRequiresGrad, name) { }

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

    public static class BatchTensorExtension
    {
        /// <summary>
        /// 勾配情報を引き継いでTensor2Dに変換
        /// </summary>
        /// <param name="batchTensor1D"></param>
        /// <returns></returns>
        public static Tensor2D ToTensor2D(this BatchTensor1D batchTensor1D, string name = "NoNameTensor2D")
        {
            var result = new Tensor2D(
                batchTensor1D.Data,
                batchTensor1D.Shape[0],
                batchTensor1D.Shape[1],
                isRequiresGrad: batchTensor1D.IsRequiresGrad,
                name: name
            );

            // 勾配計算が必要な場合、Backward関数を設定
            if (batchTensor1D.IsRequiresGrad)
            {
                result.GradInfo.Parents = new List<Tensor> { batchTensor1D };
                result.GradInfo.BackwardFn = (Tensor gradOutput) =>
                {
                    // gradOutput.Grad は Tensor2D の勾配
                    if (gradOutput.GradInfo.Grad != null)
                    {
                        // 元のBatchTensor1Dに勾配を伝播
                        batchTensor1D.GradInfo.Grad ??= Tensor.ZerosLike(batchTensor1D);

                        // 形状は同じなので、直接データをコピー
                        for (int i = 0; i < batchTensor1D.Data.Length; i++)
                        {
                            batchTensor1D.GradInfo.Grad.Data[i] += gradOutput.GradInfo.Grad.Data[i];
                        }
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// 勾配情報を引き継いでTensor2Dに変換
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor2D ToTensor2D(this Tensor tensor, string name = "NoNameTensor2D")
        {
            if (tensor.Shape.Length != 2) throw new InvalidOperationException("ToTensor2D requires shape length == 2");

            var result = new Tensor2D(
                tensor.Data,
                tensor.Shape[0],
                tensor.Shape[1],
                isRequiresGrad: tensor.IsRequiresGrad,
                name: name
            );

            // 勾配計算が必要な場合、Backward関数を設定
            if (tensor.IsRequiresGrad)
            {
                result.GradInfo.Parents = new List<Tensor> { tensor };
                result.GradInfo.BackwardFn = (Tensor gradOutput) =>
                {
                    // gradOutput.Grad は Tensor2D の勾配
                    if (gradOutput.GradInfo.Grad != null)
                    {
                        // 元のBatchTensor1Dに勾配を伝播
                        tensor.GradInfo.Grad ??= Tensor.ZerosLike(tensor);

                        // 形状は同じなので、直接データをコピー
                        for (int i = 0; i < tensor.Data.Length; i++)
                        {
                            tensor.GradInfo.Grad.Data[i] += gradOutput.GradInfo.Grad.Data[i];
                        }
                    }
                };
            }

            return result;
        }

        /// <summary>
        /// 勾配情報を引き継いでBatchTensor1Dに変換
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static BatchTensor1D ToBatchTensor1D(this Tensor tensor)
        {
            var result = new BatchTensor1D(tensor, name: "BatchTensor1D");

            // 勾配計算が必要な場合、Backward関数を設定
            if (tensor.IsRequiresGrad)
            {
                result.GradInfo.Parents = new List<Tensor> { tensor };
                result.GradInfo.BackwardFn = (Tensor gradOutput) =>
                {
                    // gradOutput.Grad は BatchTensor1D の勾配
                    if (gradOutput.GradInfo.Grad != null)
                    {
                        // 元のTensorに勾配を伝播
                        tensor.GradInfo.Grad ??= Tensor.ZerosLike(tensor);

                        // 形状は同じなので、直接データをコピー
                        for (int i = 0; i < tensor.Data.Length; i++)
                        {
                            tensor.GradInfo.Grad.Data[i] += gradOutput.GradInfo.Grad.Data[i];
                        }
                    }
                };
            }

            return result;
        }
    }

    /// <summary>
    /// BatchTensor1D: shape [B, C]
    /// </summary>
    public class BatchTensor1D : BatchTensor
    {
        /// <summary>
        /// 勾配情報を初期化して生成
        /// </summary>
        /// <param name="data"></param>
        /// <param name="batchSize"></param>
        /// <param name="features"></param>
        /// <param name="isRequiresGrad"></param>
        /// <param name="name"></param>
        public BatchTensor1D(float[] data, int batchSize, int features, bool isRequiresGrad = false, string name = "NoNameBatchTensor1D")
            : base(data, new int[] { batchSize, features }, isRequiresGrad, name) { }

        /// <summary>
        /// 勾配情報を引き継いで生成
        /// </summary>
        /// <param name="t"></param>
        /// <param name="name"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public BatchTensor1D(Tensor t, string name = "NoNameBatchTensor1D") : base(t, name)
        {
            if (t.Shape.Length != 2) throw new InvalidOperationException("BatchTensor1D requires shape length == 2");
        }

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
        /// <summary>
        /// 勾配情報を初期化して生成
        /// </summary>
        /// <param name="data"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="isRequiresGrad"></param>
        /// <param name="name"></param>
        public BatchTensor3D(float[] data, int b, int c, int h, int w, bool isRequiresGrad = false, string name = "NoNameBatchTensor3D") : base(data, new int[] { b, c, h, w }, isRequiresGrad, name) { }
        /// <summary>
        /// 勾配情報を引き継いで生成
        /// </summary>
        /// <param name="t"></param>
        /// <param name="name"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public BatchTensor3D(Tensor t, string name = "NoNameBatchTensor3D") : base(t, name)
        {
            if (t.Shape.Length != 4) throw new InvalidOperationException("BatchTensor3D requires shape length == 4");
        }
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