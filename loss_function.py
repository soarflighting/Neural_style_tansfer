'''
内容损失和风格损失
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# 内容损失通常使用回归的均方误差
class ContentLoss(nn.Module):

    def __init__(self,target,):
        super(ContentLoss,self).__init__()
        #### 我们从用于动态计算梯度的树中“分离”目标
        #### 这是一个声明的值，而不是变量
        ###将target的梯度进行分离
        self.target = target.detach()

    def forward(self,input):
        self.loss = F.mse_loss(input,self.target)

        return input


# Gram Matrix实际上可看做是feature之间的偏心协方差矩阵（即没有减去均值的协方差矩阵）
def gram_matrix(input):

    #### 特征映射
    #### a = batch (size=1)
    #### b = number
    #### (c,d) = dimensions of input
    a,b,c,d = input.size()

    #### resize F_XL into Fxl(hat)
    features = input.view(a*b,c*d)

    #### 计算gram矩阵
    G = torch.mm(features,features.t())

    #### 我们通过除以每个特征映射中的元素来“标准化”gram矩阵的值
    return G.div(a*b*c*d)

# 风格损失先生成 gram 矩阵，再使用均方误差计算损失
class StyleLoss(nn.Module):

    def __init__(self,target_feature):
        super(StyleLoss,self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self,input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G,self.target)

        return input




