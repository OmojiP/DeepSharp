namespace DeepSharp.Optimizer
{
    public class SGD
    {
        List<Tensor> parameters;
        float lr;

        public SGD(List<Tensor> parameters, float lr = 0.01f)
        {
            this.parameters = parameters;
            this.lr = lr;
        }

        public void Step()
        {
            foreach (var p in parameters)
            {
                if (p.GradInfo.Grad != null)
                {
                    for (int i = 0; i < p.Data.Length; i++)
                        p.Data[i] -= lr * p.GradInfo.Grad.Data[i];
                }
            }
        }

        public void ZeroGrad()
        {
            foreach (var p in parameters)
            {
                // null ではなく 0 のテンソルを割り当てる（既にあればゼロクリア）
                if (p.GradInfo.Grad == null)
                    p.GradInfo.Grad = Tensor.ZerosLike(p);
                else
                    for (int i = 0; i < p.GradInfo.Grad.Data.Length; i++)
                        p.GradInfo.Grad.Data[i] = 0f;
            }
        }
    }

    public class Adam
    {
        List<Tensor> parameters;
        float lr;
        float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
        Dictionary<Tensor, float[]> m = new(), v = new();
        int t = 0;

        public Adam(List<Tensor> parameters, float lr = 0.001f)
        {
            this.parameters = parameters; this.lr = lr;
            foreach (var p in parameters)
            {
                m[p] = new float[p.Data.Length];
                v[p] = new float[p.Data.Length];
            }
        }

        public void Step()
        {
            t++;
            foreach (var p in parameters)
            {
                if (p.GradInfo.Grad == null) continue;
                var grad = p.GradInfo.Grad.Data;
                var mp = m[p]; var vp = v[p];
                for (int i = 0; i < grad.Length; i++)
                {
                    mp[i] = beta1 * mp[i] + (1 - beta1) * grad[i];
                    vp[i] = beta2 * vp[i] + (1 - beta2) * grad[i] * grad[i];
                    float mHat = mp[i] / (1 - MathF.Pow(beta1, t));
                    float vHat = vp[i] / (1 - MathF.Pow(beta2, t));
                    p.Data[i] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
                }
            }
        }

        public void ZeroGrad()
        {
            foreach (var p in parameters)
            {
                if (p.GradInfo.Grad == null) p.GradInfo.Grad = Tensor.ZerosLike(p);
                else for (int i = 0; i < p.GradInfo.Grad.Data.Length; i++) p.GradInfo.Grad.Data[i] = 0f;
            }
        }
    }
}
