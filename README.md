# Open Optics Library (OpenOL)
## 目次 (table of contents)
* **[概要(abstract)](#概要)**
* **[フォルダ構成](#フォルダ構成)**
* **[依存ライブラリなど](#依存ライブラリなど)**
* **[環境構築 (編集中)](#環境構築-編集中)**  
	* [Ubuntu](#ubuntu)
* **[TODO](#todo)**

## 概要
光の回折・干渉シミュレーション及びCGH (Conputer Generated Hologram) を生成するためのプログラムです。
OpenMPとCUDAにより並列計算しています。

## フォルダ構成
- include    
  頭文字がgのものはCUDAに関するファイル
- openoltests  
  テストスクリプト  
- 3ddata  
  物体点データ
- out  
  出力画像保存用フォルダ

<a id="depend-library"></a>
## 依存ライブラリなど
- OpenCV
- FFTW3  
  fftw3とfftw3fが必要となります。
- OpenMP
- CUDA

<a id="enviroment"></a>
## 環境構築 (編集中)
### Ubuntu
1. ホームディレクトリに.openolフォルダを作成する。
	```sh
	mkdir ~/.openol
	```
2. FFTW3のインストール
	```sh
	cd ~/.openol
	mkdir fftw
	wget https://www.fftw.org/fftw-3.3.10.tar.gz
	tar -zxvf fftw-3.3.10.tar.gz
	mkdir build
	cd build
	# float and thread
	cmake -DENABLE_THREADS=ON -DENABLE_FLOAT=ON \
	-DENABLE_OPENMP=ON \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_INSTALL_LIBDIR=~/.openol/fftw/lib \
	-DCMAKE_INSTALL_BINDIR=~/.openol/fftw/bin \
	-DCMAKE_INSTALL_INCLUDEDIR=~/.openol/fftw/include ..

	make 
	make install

	# double and thread
	cmake -DENABLE_THREADS=ON \
	-DENABLE_OPENMP=ON \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_INSTALL_LIBDIR=~/.openol/fftw/lib \
	-DCMAKE_INSTALL_BINDIR=~/.openol/fftw/bin \
	-DCMAKE_INSTALL_INCLUDEDIR=~/.openol/fftw/include ..

	make 
	make install
	```
3. OPenCVのインストール
	```
	cd ~/.openol
	mkdir opencv4
	wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
	unzip opencv.zip
	# Create build directory
	mkdir -p build && cd build
	# Configure
	cmake  ../opencv-master -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_INSTALL_PREFIX=~/.openol/opencv4
	# Build
	cmake --build .
	make install
	```

# TODO
- ヘッダーオンリーでコンパイルに時間がかかるので、srcフォルダに分ける。  
  そのためにクラステンプレートの型制限する必要あり。
- 環境構築用のシェルスクリプト作成
