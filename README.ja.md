# avmosaic

画像・動画から局部を検出し、自動でモザイクをかけるツールです。コマンドライン（CLI）と Python から使え、Windows・macOS・Linux で同じように動作します。

![元画像 · 検出ボックス（--draw-boxes） · モザイク](docs/demo.jpg)

*デモはパブリックドメインの NASA 写真に `faces` プリセットを適用したものです。デフォルトのプリセットは性器と肛門を対象にします。*

[English](README.md) · [한국어](README.ko.md) · **日本語**

> **自動検出は補助であり、保証ではありません。** 特殊な角度・接写・遮蔽・照明によって検出漏れが起きることも、関係のない場所にモザイクがかかることもあります。公開する前に必ず人の目で確認してください。コンテンツに適用される法令を守る責任は利用者にあります。処理する権利のある素材にのみ使用してください。

## 特長

- オープンソースの [NudeNet](https://github.com/notAI-tech/NudeNet) v3 ONNX モデルで検出します（`320n` は同梱、より高精度な `640m` は必要なときにダウンロード）。CPU のほか、ONNX Runtime 経由で CUDA・DirectML・CoreML でも動作します。
- モザイクのタイルは固定グリッドに揃えるため、フレーム間でちらつきません。タイルの大きさは、よく引用される目安「画像の長辺の 1/100 以上、かつ 4px 以上」が初期値です。ぼかし・塗りつぶしも選べます。
- 動画ではオプティカルフローで部位を追跡するため、モザイクが移動経路に沿って伸びず、部位と一緒に動きます。検出と検出の間のフレーム、検出が一時的に途切れた区間、最初に検出される直前の数フレームまでカバーします。シーンが切り替わると追跡を打ち切ります。音声はそのまま残します。
- 対象クラス・しきい値・モデル・余白・タイルサイズ・形状・保持フレーム数・コーデック・画質など、主要なパラメータはすべてオプションで調整できます。
- フォルダの一括処理、全検出結果の JSON レポート、調整用の `--draw-boxes` モードに対応しています。
- 日本語・韓国語などのファイル名、アルファチャンネル、奇数サイズの動画、NTSC フレームレート（29.97 など）を扱えます。

## インストール

Python 3.10 以上が必要です。ffmpeg は `imageio-ffmpeg` に含まれているため、別途インストールは不要です。Intel Mac の場合、最近の ONNX Runtime は Intel Mac 向けのパッケージを配布していないため、お使いの Python のバージョンに合う古い ONNX Runtime がある場合にのみインストールできます（Intel Mac の Python 3.14 向けはありません）。

```bash
git clone https://github.com/fattaillabs/avmosaic && cd avmosaic
pip install .            # 開発用: pip install -e ".[dev]"
```

GPU（任意）：`onnxruntime` の代わりに `onnxruntime-gpu`（NVIDIA CUDA）または `onnxruntime-directml`（Windows、GPU 全般）をインストールし、`--device cuda` / `--device directml` を指定します。デフォルトの `--device auto` は CUDA → DirectML → CoreML（Apple）→ CPU の順に選びます。

## クイックスタート

```bash
avmosaic video.mp4                    # -> video_censored.mp4
avmosaic photo.jpg -o out.jpg
avmosaic ./scenes -o ./censored       # フォルダごと（サブフォルダ構成を維持）
avmosaic video.mp4 --model 640m --detect-every 2 --hold 8
avmosaic photo.jpg --draw-boxes       # 何をどのスコアで検出したかを確認
avmosaic detect video.mp4 -o detections.json   # 検出のみ（出力動画なし）
avmosaic classes                      # クラスとプリセットの一覧
avmosaic models download 640m         # 大きいモデルを事前にダウンロード
avmosaic frames sample.mp4 -o ./frames --every 2   # 動画 → ラベル付け用の画像
avmosaic label ./frames                # ブラウザでモザイクを入れるべき場所を指定
avmosaic eval regions ./frames         # 指定した場所にモザイクが入っているか確認
```

`avmosaic ファイル …` は `avmosaic censor ファイル …` の省略形です。

## パラメータ

| オプション | 初期値 | 説明 |
|---|---|---|
| `--classes` | `genitals` | プリセットまたはクラス名（カンマ区切り）。プリセット：`genitals`（女性器・男性器・肛門）、`breasts`、`buttocks`、`nudity`、`faces`、`all` |
| `--threshold` | `0.25` | 最低検出スコア。下げるほど多くかかります |
| `--model` | `320n` | `320n`、`640m`、または `.onnx` ファイルのパス |
| `--resolution` | モデル既定 | モデルの入力サイズ（32 の倍数） |
| `--device` | `auto` | `auto`、`cpu`、`cuda`、`tensorrt`、`directml`、`coreml`、`openvino` |
| `--channels` | `rgb` | モデルに渡すチャンネル順（`bgr` で本家 NudeNet と完全に同じ結果） |
| `--method` | `mosaic` | `mosaic`、`blur`、`fill` |
| `--block-size` | `auto` | モザイクのタイル：`auto`、ピクセル数（`24`）、長辺に対する割合（`2%`） |
| `--shape` | `rect` | `rect` または `ellipse`（楕円でもボックス全体を覆います） |
| `--padding` | `0.15` | ボックスを各辺でその大きさのこの割合だけ広げます |
| `--fill-color` | `black` | `--method fill` の色：`black`、`white`、`#rrggbb`、`r,g,b` |
| `--detect-every` | `1` | 動画：N フレームごとに検出（シーン切り替え時も検出）。間のフレームは追跡で補います |
| `--tracking` | `on` | 動画：オプティカルフローで追跡。`off` は `--hold` フレーム以内のボックスの和集合でかけます（単純ですが速い動きでは伸びます） |
| `--hold` | `5` | 追跡 on：最初の検出の N フレーム前までさかのぼって追跡。off：前後 N フレーム以内のボックスもかけます |
| `--max-lost` | `15` | 追跡 on：検出が途切れてから N フレームの間は追跡を続けます |
| `--smooth` | `2` | 追跡 on：ボックス位置を前後 N フレームで平均（元のボックスも必ず覆います） |
| `--codec` / `--crf` / `--preset` | `libx264` / `18` / `medium` | 動画エンコーダ設定。追加の引数は `--ffmpeg-args "…"` |
| `--no-audio` | off | 音声を削除。通常はコピーし、MP4 に合わないコーデック（PCM など）は AAC に変換します |
| `--report ファイル` | — | すべての検出結果（動画はフレームごと）を JSON に保存 |
| `--draw-boxes` | off | 出力にボックス・クラス・スコアを描画 |
| `--progress` | `auto` | 進捗表示：`auto` はターミナルでバー表示、`always` はログにも 10 秒ごとに 1 行 |
| `-o`、`--suffix`、`-y` | — | 出力ファイル/フォルダ、`-o` がないときの接尾辞、上書き |

### 調整の手順

1. `avmosaic sample.mp4 --draw-boxes --report r.json` を実行して結果を見ます。
2. 検出漏れがある → `--threshold` を下げる（例：`0.15`）、`--model 640m` を試す、`--hold` と `--padding` を上げる。
3. かかりすぎる → `--threshold` を上げ、レポートでどのクラスが検出されたか確認します。

## 追跡のしくみ

検出された部位ごとにトラックを作り、median-flow 方式のオプティカルフロー（ボックス内の格子点を Lucas–Kanade で追跡し、前後方向の誤差で外れた点を除外）でフレームごとに追いかけます。かけ漏れより、かけすぎる側に倒す設計です。

- **検出と検出の間**（`--detect-every N`）：次の検出が来たら、たまったずれを間のフレームに分配して補正します。フローの経路と補正後の経路の両方を覆います。
- **検出が途切れた区間**：最大 `--max-lost` フレームまでフローで追跡を続けます。フローも追えない場合（遮蔽・ぶれ・模様のない領域）は「止まった」と「直前の速度で動き続けた」の両方を覆います。その間に再検出されれば、途切れた区間のフレームはまだバッファにあるため、2 つの検出を結ぶ直線経路も覆います。検出のないフレームではボックスを 1% ずつ（最大 +50%）広げます。
- **最初の検出の前**：新しいトラックは `--hold` フレームさかのぼって覆います。
- **シーン切り替え**：すべてのトラックを終了し、そのフレームで検出し直します。前のシーンのボックスは持ち越しません。

## モザイクが正しい位置にあるか確認する

検閲ツールで重要なのは「ポルノかどうか」ではなく「隠すべき場所にモザイクがあるか」です。数枚のサンプルに一度だけ人が印を付けておけば、その後はコードや設定を変えるたびに自動で確認できます。

1. `avmosaic frames sample.mp4 -o ./frames --every 2` で 2 秒ごとにフレームを画像として保存します。
2. `avmosaic label ./frames` でローカルのページ（127.0.0.1 のみ）が開きます。隠すべき部位をドラッグで囲み、隠す場所がない画像は **C** を押します。ボックスは画像の隣に `<名前>.json` として保存されます。最初は空の状態から始まります（`--suggest` で検出結果を下書きにできますが、ラベルが検出器寄りになります）。
3. `avmosaic eval regions ./frames` がテキストで報告します。
   - 印を付けた部位が何 % 覆われたか（悪い順に一覧）
   - モザイクが印の外にどれだけはみ出したか
   - 隠す場所のない画像に誤ってかかっていないか

## 実際の映像で試す

avmosaic は露骨な素材を同梱・ホスト・配布しておらず、テストでも使用していません。実際の映像で試す場合は、figshare で公開されている [The Image and video dataset for adult content detection in image and video](https://figshare.com/articles/dataset/The_Image_and_video_dataset_for_adult_content_detection_in_image_and_video/14495367?file=27771210) などの研究用データセットを利用できます。

> **18 歳以上・自己責任。** このようなデータセットには露骨な成人向けコンテンツが含まれます。ダウンロードする前に、データセットのページにあるライセンスと免責事項を読んでください。ご自身の責任で、合法な範囲でのみ利用し、そのファイルをこのリポジトリにコミットしたり Issue に添付したりしないでください。

```bash
avmosaic "動画.mp4" --draw-boxes --report report.json   # 結果を自分の目で確認
avmosaic frames "動画.mp4" -o ./frames --every 2         # ラベル付け + 位置の確認
```

## 速度

Apple M1（CoreML）で、1080p・30fps・20 秒（600 フレーム）の動画で計測しました。

| 設定 | 時間 | 処理速度 |
|---|---|---|
| 初期設定（`320n`） | 22.0 秒 | 27fps |
| `--detect-every 2` | 16.1 秒 | 37fps |
| `--model 640m --detect-every 3` | 75.6 秒 | 7.9fps |

## Python API

```python
from avmosaic import CensorOptions, VideoOptions, censor_image, censor_video, create_detector

detector = create_detector("320n", classes=["FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"],
                           threshold=0.2)
censor_image("in.jpg", "out.jpg", detector, CensorOptions(block_size="2%"))
censor_video("in.mp4", "out.mp4", detector, CensorOptions(), VideoOptions(hold=8))
```

`detect(bgr_image) -> list[Detection]` メソッドを持つオブジェクトなら何でも検出器として使えます。独自に学習したモデルに差し替えることもできます。

## テスト

```bash
pip install -e ".[dev]"
pytest                    # すべて（約 10 秒）
pytest -m "not model"     # 実際のモデルを使うテストを除外
```

テストに露骨な素材は一切使っていません。合成した赤黒のチェッカー模様を「隠すべき部位」とみなす色検出器、メモリ上で作るフレーム、パブリックドメイン写真の顔クラスで、パイプライン・追跡・実モデルを検証します。CI は Linux・macOS・Windows で Python 3.10、3.12、3.14 を使ってテストを実行します。

## 制限事項

- 誤りの主な原因は検出器です。NudeNet は一般的な露出画像で学習されているため、大きな画面の中の小さな部位、極端な接写、部分的な遮蔽は苦手です。実写向けのモデルで、アニメ・イラストには対応していません。
- 可変フレームレート（VFR）の動画は固定フレームレートに変換されます（音ズレ防止のため）。
- 字幕・データストリームはコピーされません。奇数サイズの動画は H.264 のため 1px 追加します。
- 画像のメタデータ（EXIF）は出力にコピーされません。EXIF の向き情報は先に反映します。

## ライセンス

本プロジェクトのコードは MIT です（[LICENSE](LICENSE)）。NudeNet のライセンスに関する重要な注意を含むサードパーティ情報は [NOTICE](NOTICE) を参照してください。
