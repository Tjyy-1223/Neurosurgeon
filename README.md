# Neurosurgeon
🥳本项目根据经典论文：Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge进行实现，为DNN模型选取划分点后分别部署在云端设备和边端设备上进行协同推理(Collabrative Inference)。

论文链接🔗：[Collaborative Intelligence Between the Cloud and Mobile Edge.pdf](https://github.com/Tjyy-1223/Neurosurgeon/blob/main/paper/Collaborative Intelligence Between the Cloud and Mobile Edge.pdf)

![image-20230524094940267.png](https://github.com/Tjyy-1223/Neurosurgeon/blob/main/assets/image-20230524094940267.png?raw=true)

具体工作：

1）初步使用四种经典的DNN模型进行构建。

2）DNN模型层级特征研究：Layer latency 和 Size of output data。

2）Deployment Phase：在本地机器运行DNN层得到构建预测模型，提供本地数据集和模型参数。

3）Runtime Phase：实现DNN模型协同推理，具体脚本命令参考下面的描述。



#### 项目结构

+ models：实现四种DNN模型 - LeNet、AlexNet、VggNet-16、MobileNet-v2
+ utils: 
  + inference_utils: 与DNN模型推理相关的功能函数
  + excel_utils: 与excel表存储和读取相关的功能函数
+ predictor：
  + 



#### 项目运行







#### 总结

Neurosurgeon是云边协同推理中的优秀框架，首次实现了将DNN模型部署在云边端设备进行协同推理。

但其也有相应的局限性：

+ 适用于链式拓扑结构
+ 没有考虑DNN块中的划分点
+ 只考虑了静态网络环境下的划分状况
