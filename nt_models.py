'''
建立模型
'''
import torch
import torch.nn as nn
import torchvision.models as models
import copy
from loss_function import ContentLoss,StyleLoss
from nt_utils import image_loader



class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std



class NT_models(nn.Module):

    def __init__(self):
        super(NT_models,self).__init__()
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5']

        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        self.vgg19_features = models.vgg19(pretrained=True).features.eval()



    def get_style_content_model_and_loss(self,style_img,content_img):

        normalization = Normalization(self.cnn_normalization_mean,self.cnn_normalization_std)
        cnn = copy.deepcopy(self.vgg19_features)
        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():

            ### 如果layer 匹配conv2d
            if isinstance(layer,nn.Conv2d):
                i+=1
                name = "conv_{}".format(i)
            elif isinstance(layer,nn.ReLU):
                name = 'relu_{}'.format(i)
                #### 对于我们在下面插入的‘ContentLoss’和‘StyleLoss’
                #### 本地版本不能很好的发挥作用，我们在这里替换不合适的
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer,nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer,nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer :{}'.format(layer.__class__.__name__))

            model.add_module(name,layer)

            if name in self.content_layers:

                ###add the content loss
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i),content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                ### add style loss
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i),style_loss)
                style_losses.append(style_loss)

        ## 现在我们在最后的内容和风格损失之后减掉了图层
        for i in range(len(model)-1,-1,-1):
            if isinstance(model[i],ContentLoss) or isinstance(model[i],StyleLoss):
                break

        model = model[:(i+1)]

        return model,style_losses,content_losses




if __name__ == '__main__':
    style_img = image_loader("c:/Users/Mr.fei/pytorch-learn/data/images/picasso.jpg")
    content_img = image_loader("c:/Users/Mr.fei/pytorch-learn/data/images/dance.jpg")
    nt_model = NT_models()
    model,content_losses,style_losses = nt_model.get_style_content_model_and_loss(style_img=style_img,content_img=content_img)

    print(model)
    print(content_losses)
    print(style_losses)


