# Open Optics Library (OpenOL)
# 概要
光の回折・干渉シミュレーション及びCGH (Conputer Generated Hologram) を生成するためのプログラムです。
OpenMPとCUDAにより並列計算しています。

# フォルダ構成
- include    
  頭文字がgのものはCUDAに関するファイル
- openoltests  
  テストスクリプト  
- 3ddata  
  物体点データ
- out  
  出力画像保存用フォルダ
# 依存ライブラリなど
- OpenCV
- FFTW3  
  fftw3とfftw3fが必要となります。
- OpenMP
- CUDA

# 環境構築


# TODO
- ヘッダーオンリーでコンパイルに時間がかかるので、srcフォルダに分ける。  
  そのためにクラステンプレートの型制限する必要あり。
