# DeepSharp

# 概要

C#で深層学習するためのライブラリ(制作中)

Kaggleで配布されているmnistのCSV[データセット](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)を
分類するMLPをCPUで学習するサンプルが入っています

# 使用方法

## 共通部分

- データセットをダウンロードして任意の場所に配置
- リポジトリをclone
- Program.csのデータセットのパスを書き換える


## Windows

VisualStudio2022で開いて実行

## Mac / Linux

`生成AIに書かせました`

### .NET SDKのインストール

**Mac**

```bash
# Homebrewを使用する場合
brew install dotnet

# または公式インストーラーを使用
# https://dotnet.microsoft.com/download からダウンロード
```


**Ubuntu/Debian**

```bash
# Microsoft パッケージリポジトリを追加
wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb

# .NET SDKをインストール
sudo apt-get update
sudo apt-get install -y dotnet-sdk-9.0
```

**CentOS/RHEL/Fedora**

```bash
sudo dnf install dotnet-sdk-9.0
```

### プロジェクトの実行

インストール後、以下のコマンドでslnファイルを操作

```bash
# プロジェクトディレクトリに移動
cd {DeepSharp-Directory}

# 依存関係の復元
dotnet restore

# ビルド
dotnet build

# 実行
dotnet run
```



# コードの説明

PyTorchを踏襲して型情報を加えたイメージ

## 自動微分


$$y = a \times b$$

$$\frac{\partial L}{\partial y}$$　を upstream gradient としてBackwardFnから受け取る

受け取った上流の勾配を利用して下流の勾配を計算する
$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial y} \times b^T$$
$$\frac{\partial L}{\partial b} = a^T \times \frac{\partial L}{\partial y}$$

他の変数が絡む場合は足し合わせる必要があるので、`AddInto(b.GradInfo.Grad, gradB);`

```cs
/// <summary>
/// y = a * b, 
/// a: [M, N], b: [N, P] -> y: [M, P]
/// </summary>
public static Tensor2D MatMul(Tensor2D a, Tensor2D b, bool isRequiresGrad = true)
{
    float[] resultData = MultiplyMatrix(a.Data, a.Shape, b.Data, b.Shape);
    var result = new Tensor2D(resultData, a.Shape[0], b.Shape[1], isRequiresGrad: a.IsRequiresGrad || b.IsRequiresGrad, name: "MatMul");

    if (isRequiresGrad && (a.IsRequiresGrad || b.IsRequiresGrad))
    {
        result.GradInfo.Parents = new List<Tensor> { a, b };
        result.GradInfo.BackwardFn = (Tensor dLdResult) =>
        {
            if (dLdResult == null)
                throw new InvalidOperationException("Gradient output is null in MatMul backward function.");

            var dLdResultTensor2D = dLdResult.ToTensor2D();
            
            // gradOutput.Grad は [B, P]
            if (a.IsRequiresGrad)
            {
                // dL/da = dL/dy * b^T
                var gradA = MatMul(dLdResultTensor2D, Transpose(b), isRequiresGrad: false);
                // dL/da を a.Grad に加算
                a.GradInfo.Grad ??= ZerosLike(a);
                AddInto(a.GradInfo.Grad, gradA);
            }
            if (b.IsRequiresGrad)
            {
                // dL/db = a^T * dL/dy
                var gradB = MatMul(Transpose(a), dLdResultTensor2D, isRequiresGrad: false);
                // dL/db を b.Grad に加算
                b.GradInfo.Grad ??= ZerosLike(b);
                AddInto(b.GradInfo.Grad, gradB);
            }
        };
    }
    return result;
}
```


## Program.cs

学習ループ

## Dataset

## DataLoder

## Model

## Layer

## Func

## Loss

## Tensor

```cs
public class Tensor
{
        public float[] Data;
        public readonly int[] Shape;

        public GradInfo GradInfo;
        public bool IsRequiresGrad { get; set; }

        public string Name { get; set; } = "";
}

public class GradInfo
{
    public Tensor? Grad;
    public Action<Tensor>? BackwardFn;
    public List<Tensor> Parents;
}
```


```cs
public static void Backward(this Tensor tensor)
{

    // 出発点の勾配は 1（dL/dL = 1）
    if (tensor.GradInfo.Grad == null)
        tensor.GradInfo.Grad = Tensor.OnesLike(tensor);

    // トポロジカル順に並べる
    var topo = Tensor.TopologicalSort(tensor);

    // 後ろから順にBackward
    foreach (var t in topo)
    {
        //Console.WriteLine($"Backward: {t.Name}, IsRequiresGrad={t.IsRequiresGrad}, Parents={string.Join(", ", t.GradInfo.Parents.Select(p => p.Name))}");

        if (t.GradInfo.BackwardFn != null)
        {
            t.GradInfo.BackwardFn(t);
        }
    }
}
```

```cs
class ScalarTensor : Tensor
class Tensor1D : Tensor
class Tensor2D : Tensor
class Tensor3D : Tensor
class BatchTensor : Tensor
class BatchTensor1D : BatchTensor
class BatchTensor3D : BatchTensor
```

### ToTensorND

TensorをそのままTensorNDに変換してもGradInfoの関数やParentの情報をうまく伝搬できないため、
勾配情報をそのまま流すようにBackwardFn, Parentを設定したTensorNDを生成する拡張メソッド

```cs
public static class TensorConversion
{
    public static Tensor1D ToTensor1D(this Tensor tensor, string name = "NoNameTensor1D")
    public static Tensor2D ToTensor2D(this Tensor tensor, string name = "NoNameTensor2D")
    public static BatchTensor1D ToBatchTensor1D(this Tensor tensor, string name = "NoNameBatchTensor1D")
    public static Tensor2D ToTensor2D(this BatchTensor1D batchTensor1D, string name = "NoNameTensor2D")
}
```

### 計算  

```cs
public partial class Tensor
{
    public static Tensor operator +(Tensor a, Tensor b)
    public static void AddInto(Tensor dest, Tensor src)
    public static BatchTensor1D MatMul(BatchTensor1D a, Tensor2D b)
    public static Tensor1D MatMul(Tensor1D a, Tensor2D b)
    public static Tensor2D MatMul(Tensor2D a, Tensor2D b)
    public static Tensor AddWithBroadcast(Tensor a, Tensor b)

    private static float[] AddArrays(float[] a, float[] b)
    private static float[] MultiplyMatrix(float[] a, int[] ashape, float[] b, int[] bshape)
}
```

### Helper, Util

```cs
public partial class Tensor
{
    public static Tensor ZerosLike(Tensor t, string name = "NoNameTensor") => new Tensor(new float[t.Data.Length], (int[])t.Shape.Clone(), t.IsRequiresGrad, name);
    public static Tensor OnesLike(Tensor t, string name = "NoNameTensor") => new Tensor(Enumerable.Repeat(1f, t.Data.Length).ToArray(), (int[])t.Shape.Clone(), t.IsRequiresGrad, name);
    public static Tensor FullLike(Tensor t, float value, string name = "NoNameTensor") => new Tensor(Enumerable.Repeat(value, t.Data.Length).ToArray(), (int[])t.Shape.Clone(), t.IsRequiresGrad, name);
}
```

```cs
public partial class Tensor
{
    public static List<Tensor> TopologicalSort(Tensor root)
    private static int[] ComputeStrides(int[] shape)
    public static Tensor2D Transpose(Tensor2D t)
    public static Tensor Transpose(Tensor t, int[]? perm = null)
    public static void ClipGrad(Model.Model model, float maxNorm)
    public static int[] ConcatShape(int batchSize, int[] sampleShape)
}
```
