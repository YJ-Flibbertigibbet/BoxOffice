# 模型说明
* 测试使用数据集：box_office/data/raw_ceshi.xlsx
## 主要模型
>> resnet18+fc : res_fc文件夹
* 在config文件夹中修改实验参数
* 调用海报与剧照进行训练

>> Catboost
* 仅仅使用份额份分类与数值变量进行预测

>> mulModel1
* 将res_fc训练出来的模型对整个数据集进行预测
* 增加一列特征：用res_fc得出的票房预测结果
* 利用已整合出的信息进行catboost训练与预测
