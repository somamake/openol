# Open Optics Library (OpenOL)
# 概要
光の回折・干渉シミュレーション及びCGH (Conputer Generated Hologram) を生成するためのプログラムです。
OpenMPとCUDAにより並列計算しています。

# フォルダ構成
- include    
  頭文字がgのものはCUDAに関するファイル
- openoltests  
  テストスクリプト  
  動作のために物体点データ用意する必要あり。

# 依存ライブラリなど
- OpenCV
- FFTW3
- OpenMP

# TODO
- ヘッダーオンリーでコンパイルに時間がかかるので、srcフォルダに分ける。  
  そのためにクラステンプレートの型制限する必要あり。
