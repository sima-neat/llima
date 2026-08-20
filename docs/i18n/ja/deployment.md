# モデルのデプロイ

## 概要

コンパイル後、モデルを実行するには、Modalix デバイスにデプロイする必要があります。Model Compiler は、このプロセスを効率化するためのユーティリティである `llima-deploy` を提供します。

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy <source_directory> <destination_directory>
```

場所：

- `source_directory` - コンパイルされたモデルのディレクトリへのパス（`devkit/`、`mpk/`、およびオプションで`npy_files/`のサブディレクトリを含む`sima_files/`が含まれます）。
- `destination_directory` - Modalix デバイス上のターゲットディレクトリ（または、rsync を使用したデプロイメントの場合のローカルパス）。

このコマンドを実行すると、デプロイツールは以下の3つの主要なステップを実行します。

1.  ソースディレクトリに、必要なファイル（`sima_files/devkit/`と`sima_files/mpk/`）が含まれていることを**検証**します。
2.  **抽出** MPK アーカイブから ELF ファイルを抽出します (`*.tar.gz`)
3.  次のファイルを、`rsync` を使用して宛先に同期します。
    - `devkit/` - ランタイムのオーケストレーションファイル
    - `elf_files/` - 抽出されたバイナリファイル
    - `npy_files/` - LoRAアダプターの重み（存在する場合、自動的に含まれます）。

このツールは、効率的なファイル転送のために内部的に`rsync`を使用し、すでに最新バージョンのファイルは転送をスキップします。

## デプロイのワークフロー

`llima-compile` を使用してモデルをコンパイルすると、次のようなディレクトリ構造になります。

``` text
Llama-3.2-3B-Instruct_out/
├── onnx_files/
└── sima_files/
    ├── devkit/
    └── mpk/
```

このソフトウェアをModalixデバイスにデプロイするには、次の2つの方法があります。

**方法A：Modalixデバイスへの直接デプロイ**

ホストマシンがModalixデバイスにネットワーク経由でアクセスできる場合：

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy Llama-3.2-3B-Instruct_out sima@192.168.1.20:/media/nvme/llima/llama3_2
```

**オプションB：ローカルディレクトリに展開し、手動で転送する**

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy Llama-3.2-3B-Instruct_out llama3_2
sima-user@docker-image-id:/home/docker$ scp -r llama3_2 sima@192.168.1.20:/media/nvme/llima/
```

:::note
`192.168.1.20` は、Modalix IPアドレスの例です。お使いのデバイスのIPアドレスを使用してください。
:::

展開後、Modalix デバイスに SSH で接続し、モデルを実行します。

``` console
modalix:~$ ssh sima@192.168.1.20
```

次に、`llima` CLIを使用してモデルを実行します。詳細については、[LLiMa CLI](runtime.md) を参照してください。

``` console
modalix:~$ llima run <model_name>
```

## 推測的デコーディングモデル

`llima-compile` に `--draft_model_path` を指定すると、その出力には、ターゲットとドラフトのコンパイラ出力が、1つの親ディレクトリの下にまとめられます。親ディレクトリを1つのコマンドでデプロイします。

``` console
llima-deploy compiled-eagle3 spec-decoding-output
```

デプロイされたパッケージには、通常のランタイムモデルディレクトリが 2 つ含まれています。

``` text
spec-decoding-output/
├── <target-model>/
│   ├── devkit/
│   └── elf_files/
└── <draft-model>/
    ├── devkit/
    └── elf_files/
```

親ディレクトリを実行することで、LLiMa が、それぞれのシリアライズされた推測デコーディング設定から、両方のモデルを識別して読み込めるようにします。

``` console
llima run spec-decoding-output
```

## トラブルシューティング

**エラー：「devkit ディレクトリが見つかりません」**

ソースディレクトリが、`llima-compile` の出力ディレクトリであることを確認してください。このディレクトリには、`sima_files` というサブディレクトリが含まれている必要があります。

**エラー：「mpk ディレクトリが見つかりません」**

コンパイルが正常に完了したことを確認してください。`sima_files/mpk/` ディレクトリには、`.tar.gz` ファイルが含まれている必要があります。

**デプロイに時間がかかっています**

- 圧縮機能付きの `rsync` を使用してください。このツールは、デフォルトで `rsync -aP` を使用します。
- Modalix にモデルをデプロイし、NVMe ストレージを使用することで、モデルの読み込みを高速化できます。
- 変更されたファイルのみをデプロイすることを検討してください。 `--resume` コンパイル中に
