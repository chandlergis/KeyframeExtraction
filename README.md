# KeyframeExtraction
Pythonでキーフレーム抽出

## アルゴリズム
- ヒストグラム類似度 - [histgram.py](./src/histogram.py)
- block based histogram
- k-means - [kmeans.py](./src/kmeans.py)

## 環境
``` bash
python -m venv keyframeEnv
```

``` bash
// Windows
.\keyframeEnv\Scripts\activate

// Linux
source keyframeEnv/bin/activate
```

``` bash
pip install -r requirements.txt
```