# report_003：README の作成

## 概要

プロジェクトルートに `README.md` を作成した．
読者はリポジトリを初めて開く研究室のメンバーや共同研究者を想定し，プロジェクトの概要，主な結果，論文との主な違い，動作環境，セットアップ，実行方法，ファイル構成，参考文献を記載した．
各ファイルの役割と依存関係の詳細は `document.md`，実験の詳細は `.reports/report_001.md` に委ね，README には入口となる情報のみを記載した．

## 記載内容の根拠

主な結果の数値（PSNR と SSIM の 0.5，1.0，2.0 bpp の値，JPEG 2000 との差）は `.reports/report_001.md` の表4および `outputs/ex001_kodak_cae/figures/curves.csv` と一致させた．
README から参照する図 `.reports/report_001_figures/figure4.png` は Git の管理対象であり，`outputs/` が管理対象外でも GitHub 上で表示される．

## 未解決の点と次に検討すべき事項

1. `reference/theis2017.pdf`（論文 PDF）がリポジトリに含まれている．公開リポジトリにする場合は，論文 PDF の再配布の可否を確認する必要がある．
2. ライセンスは README に記載していない．公開する場合はライセンスを決めて追記する必要がある．
