namespace DeepSharp
{
    public static class TensorConversion
    {
        /// <summary>
        /// 勾配情報を引き継いだTensor1Dを生成
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static Tensor1D ToTensor1D(this Tensor tensor, string name = "NoNameTensor1D")
        {
            if (tensor.Shape.Length != 1) throw new InvalidOperationException("ToTensor1D requires shape length == 2");

            var result = new Tensor1D(
                tensor.Data,
                isRequiresGrad: tensor.IsRequiresGrad,
                name: name
            );

            // 勾配計算が必要な場合、Backward関数を設定
            if (tensor.IsRequiresGrad)
            {
                result.GradInfo.Parents = new List<Tensor> { tensor };
                result.GradInfo.BackwardFn = (Tensor gradOutput) =>
                {
                    // gradOutput.Grad は Tensor1D の勾配
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
        /// 勾配情報を引き継いだTensor2Dを生成
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
        /// 勾配情報を引き継いだBatchTensor1Dを生成
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static BatchTensor1D ToBatchTensor1D(this Tensor tensor, string name = "NoNameBatchTensor1D")
        {
            if (tensor.Shape.Length != 2) throw new InvalidOperationException("ToBatchTensor1D requires shape length == 2");

            var result = new BatchTensor1D(
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

        /// <summary>
        /// 勾配情報を引き継いだTensor2Dを生成
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
    }
}
