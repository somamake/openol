# Open Optics Library (OpenOL)
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
