namespace DeepSharp.DataLoder
{
    class MnistCsvLoader
    {
        public static List<(Tensor, int)> LoadCsv(string path)
        {
            var data = new List<(Tensor, int)>();

            foreach (var line in File.ReadLines(path).Skip(1)) // 1行目はヘッダ
            {
                var parts = line.Split(',');
                int label = int.Parse(parts[0]);

                float[] pixels = parts.Skip(1)
                                      .Select(v => float.Parse(v) / 255f) // 0〜1に正規化
                                      .ToArray();

                var tensor = new Tensor(pixels, new int[] { 1, 784 }, requiresGrad: false);
                data.Add((tensor, label));
            }

            return data;
        }

        public static IEnumerable<(Tensor, int[])> GetBatches(List<(Tensor, int)> dataset, int batchSize)
        {
            var rnd = new Random();
            var shuffled = dataset.OrderBy(x => rnd.Next()).ToList();

            for (int i = 0; i < shuffled.Count; i += batchSize)
            {
                var batch = shuffled.Skip(i).Take(batchSize).ToList();
                float[] xData = batch.SelectMany(d => d.Item1.Data).ToArray();
                int[] yData = batch.Select(d => d.Item2).ToArray();
                var xTensor = new Tensor(xData, new int[] { batch.Count, 784 }, requiresGrad: false);
                yield return (xTensor, yData);
            }
        }
    }
}
