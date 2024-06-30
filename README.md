基于本人的 https://github.com/3486038424/aniImgTagBooru 项目进行的一个小开发
可以通过提供图片文件，检索相似的图片
采用datasketch的MinHashLSHForest进行相似图片检索
而图片向量则通过打标模型删除最后一层后的输出进行计算
访问127.0.0.1:5005即可
目前未提供多种图片类型的方法，请自己修改代码使其可以支持多种图片

可以使用的模型请访问 https://github.com/3486038424/aniImgTagBooru 的release中下载
### 安装依赖
```
pip install torch
pip install Pillow
pip install datasketch
pip install flask
pip install torchvision
pip install numpy
```

`