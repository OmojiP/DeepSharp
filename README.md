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

## Tensor

