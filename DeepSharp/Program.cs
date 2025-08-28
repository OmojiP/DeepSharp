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
            var optim = new Adam(model.Parameters(), 0.01f);

            var trainData = MnistCsvLoader.LoadCsv(@"C:\Users\kodai\Downloads\mnist_test.csv\mnist_test.csv");

            var startTime = DateTime.Now;

            for (int epoch = 0; epoch < 50; epoch++)
            {
                float epochLossSum = 0f;
                int batchCount = 0;
                int correct = 0;
                int total = 0;

                foreach (var (batchX, batchY) in MnistCsvLoader.GetBatches(trainData, 32))
                {
                    var logits = model.Forward(batchX);
                    var loss = Loss.CrossEntropy(logits, batchY);

                    optim.ZeroGrad();
                    loss.Backward();

                    // debug: first layer grad norm
                    var w = model.Parameters()[0];
                    float sumsq = 0f;
                    if (w.Grad != null)
                        for (int i = 0; i < w.Grad.Data.Length; i++) sumsq += w.Grad.Data[i] * w.Grad.Data[i];
                    //Console.WriteLine($" batch grad_norm_W0: {MathF.Sqrt(sumsq)}");

                    Tensor.ClipGrad(model.Parameters(), 5.0f);
                    optim.Step();

                    epochLossSum += loss.Data[0];
                    batchCount++;

                    // accuracy (predict argmax of logits)
                    int batchSize = logits.Shape[0];
                    int classes = logits.Shape[1];
                    for (int i = 0; i < batchSize; i++)
                    {
                        int pred = 0;
                        float best = logits.Data[i * classes + 0];
                        for (int j = 1; j < classes; j++)
                        {
                            if (logits.Data[i * classes + j] > best) { best = logits.Data[i * classes + j]; pred = j; }
                        }
                        if (pred == batchY[i]) correct++;
                        total++;
                    }
                }

                Console.WriteLine($"epoch {epoch}, loss_avg {epochLossSum / batchCount}, acc {correct / (float)total:F4}");
            }

            var endTime = DateTime.Now;

            Console.WriteLine((startTime - endTime).ToString());
        }
    }
}