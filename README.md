# pedestrian_detection_using_mtcnn
## 本工程基于vs2013实现
本工程包含三部分: 训练模型，提取模型的权重，加载权重实现推理
## 训练模型 - mtcnn-master-caffe
### 准备数据
1. P-net: gen_36-12_net_data.py, gen_36-12_net_list.py
2. R-net: gen_72-24_net_data.py, gen_72-24_net_list.py
3. O-net: gen_144-48_net_data.py, gen_144-48_net_list.py
### 定义网络
1. P-net: Pnet_36-12-train.prototxt
2. R-net: Rnet_72-24-train.prototxt
3. O-net: Onet_144-48-train.prototxt
### 保存的模型
 ./mtcnn-master-caffe/myself_mtcnn/train/models/models_*-*
## 提取模型权重 - my_caffemodel_2_mtcnnmodel
只需更改为需要提取的模型地址即可
## 加载权重推理检测行人 - my_mtcnn_light
基于mtcnn_light工程稍作修改

## 参考工程
https://github.com/dlunion/mtcnn<br/>
https://github.com/AlphaQi/MTCNN-light<br/>




