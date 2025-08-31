namespace DeepSharp.DataLoder
{
    class MnistCsvLoader
    {
        public static Dataset<Tensor1D> LoadCsv(string path)
        {
            var dataset = new Dataset<Tensor1D>();

            foreach (var line in File.ReadLines(path).Skip(1)) // 1行目はヘッダ
            {
                var parts = line.Split(',');
                int label = int.Parse(parts[0]);

                float[] pixels = parts.Skip(1)
                                      .Select(v => float.Parse(v) / 255f) // 0〜1に正規化
                                      .ToArray();

                var tensor = new Tensor1D(pixels, name: "MNIST_Image");
                dataset.Add(new LabeledData<Tensor1D>(tensor, label));
            }

            return dataset;
        }

        public static IEnumerable<BatchedData<BatchTensor1D>> GetBatches(Dataset<Tensor1D> dataset, int batchSize)
        {
            var rnd = new Random();
            var shuffled = dataset.OrderBy(x => rnd.Next()).ToList();

            for (int i = 0; i < shuffled.Count; i += batchSize)
            {
                // i番目のバッチを取得
                var batch = shuffled.Skip(i).Take(batchSize).ToArray();
                // バッチのデータとラベルをまとめてTensorと配列に変換
                BatchTensor1D batchTensor = new BatchTensor1D(batch.Select(x => x.Data).ToArray(), name: "BatchTensor1D(MNIST)");
                int[] batchLabel = batch.Select(d => d.Label).ToArray();
                
                yield return new BatchedData<BatchTensor1D>(batchTensor, batchLabel);
            }
        }
    }
}
