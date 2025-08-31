using DeepSharp;
using DeepSharp.DataLoder;
using DeepSharp.Model;
using DeepSharp.Optimizer;

namespace DeepSharp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var model = new MLP();
            //var optim = new SGD(model.Parameters(), 0.01f);
            var optim = new Adam(model.GetParameters(), 0.01f);

            var trainData = MnistCsvLoader.LoadCsv(@"C:\Users\kodai\Downloads\mnist_test.csv\mnist_test.csv");

            var startTime = DateTime.Now;

            for (int epoch = 0; epoch < 50; epoch++)
            {
                float epochLossSum = 0f;
                int batchCount = 0;
                int correct = 0;
                int total = 0;

                foreach (var batch in MnistCsvLoader.GetBatches(trainData, 32))
                {
                    //Console.WriteLine($"\r epoch {epoch} batch {batchCount}");

                    var logits = model.Forward(batch.BatchTensor);
                    ScalarTensor loss = Loss.CrossEntropy(logits, batch.Labels);

                    optim.ZeroGrad();
                    loss.Backward();

                    // デバッグ: 勾配確認（最初のバッチのみ）
                    //if (batchCount == 0 && epoch < 5)
                    //{
                    //    var params1 = model.GetParameters();

                    //    // 各パラメータの勾配ノルムを確認
                    //    for (int p = 0; p < params1.Count; p++)
                    //    {
                    //        if (params1[p].GradInfo.Grad != null)
                    //        {
                    //            var gradNorm = Math.Sqrt(params1[p].GradInfo.Grad.Data.Select(x => (double)(x * x)).Sum());
                    //            Console.WriteLine($"  Param {p} gradient norm: {gradNorm:F6}");
                    //        }
                    //        else
                    //        {
                    //            Console.WriteLine($"  Param {p}: NO GRADIENT");
                    //        }
                    //    }

                    //    // logitsの勾配も確認
                    //    if (logits.GradInfo.Grad != null)
                    //    {
                    //        var logitGradNorm = Math.Sqrt(logits.GradInfo.Grad.Data.Select(x => (double)(x * x)).Sum());
                    //        Console.WriteLine($"  Logits gradient norm: {logitGradNorm:F6}");
                    //    }
                    //}

                    Tensor.ClipGrad(model, 5.0f);
                    optim.Step();

                    epochLossSum += loss.Item;
                    batchCount++;

                    // accuracy (predict argmax of logits)
                    int batchSize = logits.BatchSize;
                    int classes = logits.Features;
                    for (int i = 0; i < batchSize; i++)
                    {
                        int pred = 0;
                        float best = logits.Data[i * classes + 0];
                        for (int j = 1; j < classes; j++)
                        {
                            if (logits.Data[i * classes + j] > best) { best = logits.Data[i * classes + j]; pred = j; }
                        }
                        if (pred == batch.Labels[i]) correct++;
                        total++;
                    }
                }

                // 最初の数エポックは詳細な情報を表示
                //if (epoch < 3)
                //{
                //    var params1 = model.GetParameters();
                //    Console.WriteLine($"  Weight range: {params1[0].Data.Min():F6} to {params1[0].Data.Max():F6}");
                //}

                Console.WriteLine($"epoch {epoch}, loss_avg {epochLossSum / batchCount}, acc {correct / (float)total:F4}");
            }

            var endTime = DateTime.Now;

            Console.WriteLine((endTime - startTime).ToString());
        }
    }
}