# dev diary

## 我们的关切
- 信息安全：
- - 大数据时代，隐私越来越成为一种有价值的数据。一种易于部署，准确率高的本地邮件分类器体现出其信息安全的价值
## 主要挑战和行动
### 那些要素重要
1. 数据: 数据是知识的基础,对于连结主义的ai是关键中的关键。所以，我们找到了一个公认数据集（这里指有大量基于此数据集训练的模型，易于平行比较）并且对结构进行了分析。
2. 模型选择：不同的结构之间性能存在较大差异，所以我们参考了部分博客和论文，对不同模型进行了测试选择了cnn-lstm。
### 制定工作流程
* 分析和整理数据
* * 分析平衡性：
* * 进行数据清洗
* 定义模型
* 对超参进行优化
* 验证模型的有效性
* 训练模型并保存
* 对模型进行部署

## 零碎的挑战和解决方案
### *不平衡的数据*
展示
```
Number of all messages: 5574
Number of ham messages: 4827
Number of spam messages: 747
```
**解决方案**：超采样 spam 
- smote：插值方法
- simple over-sampling： 重复采样同一个数据
第二种方法更容易过拟合,因而选择第一种**smote**

### 过拟和
* drop out 随机改变权重
* early stopping test loss 持续上升时停止训练
## 项目结构
config == 配置  
dataset == 数据集的初始文件（SMSSpamCollection）  
           中间文件 etc  
            最终文件 cleaned_data_set_mv.pkl  
---
preprocess == 各种预处理脚本的库  
            
train == 训练时的脚本库  
- conv_lstm.py // rnnModel.py: 定义模型的结构  
- dataset
- trainer
- utils  

---
usage == 对应训练的步骤  
- preprocess ==     预处理， 依赖同名库  
- find_best find == the best hyperparameters for model  
- validate_best ==  test my model with k fold, _which depends on parameters that found by find_best_   
- final_model ==    train finally model, _which depends on parameters that found by find_best_   
---
model == a folder for saving trained model
- model/model.pth 模型本尊  
---
aRun ==       dao and services for backends?  
main ==       backend program for web ui  
main_with_experiential_ui ==   python ui  

## api
### GET request to example server
GET http://127.0.0.1:5000/emails
Content-Type: application/json
### response
HTTP/1.1 200 OK
Server: Werkzeug/3.1.3 Python/3.12.7
Date: Sun, 08 Dec 2024 07:18:30 GMT
Content-Type: application/json
Content-Length: 213
Connection: close

[
  {
    "Label": "spam",
    "Message": "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info",
    "prediction": "spam"
  }
]


### POST request to example server
POST http://127.0.0.1:5000/emails
Content-Type: application/json

{
  "content": [
    "msg1",
    "msg2",
    "07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow"
  ]
}

### response
HTTP/1.1 200 OK
Server: Werkzeug/3.1.3 Python/3.12.7
Date: Sun, 08 Dec 2024 07:18:30 GMT
Content-Type: application/json
Content-Length: 27
Connection: close

{
  "spams": [
    2
  ]
}
###


