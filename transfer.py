'''
转换函数
'''
import torch.optim as optim
from nt_models import NT_models
from nt_utils import image_loader,imshow

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def transfer(content_img,style_img,input_img,num_steps = 50,style_weight=1000000,content_weight = 1):

     print('Building the style transfer model...')
     nt_model = NT_models()
     model,style_losses,content_losses = nt_model.get_style_content_model_and_loss(style_img,content_img)

     optimizer = get_input_optimizer(input_img)
     print('Optimizing ... ')

     run = [0]

     while run[0] <= num_steps:

        def closure():
            #### 更正更新的输入图像的值
            #### 使用clamp_() 将图像的值改变为0-1之间
            input_img.data.clamp_(0,1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score+= sl.loss

            for cl in content_losses:
                content_score+=cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score+content_score
            print("loss = ",loss)
            loss.backward()

            run[0] += 1

            if run[0]%50 == 0:
                print("Run {}:".format(run))
                print("Style Loss :{:.4f} Content Loss :{:.4f}".format(style_score.item(),content_score.item()))

                print()

            return style_score+content_score

        optimizer.step(closure)

    ## a last correction
     input_img.data.clamp_(0,1)

     return input_img.detach()




if __name__ == '__main__':
    style_img = image_loader("c:/Users/Mr.fei/pytorch-learn/data/images/picasso.jpg")
    content_img = image_loader("c:/Users/Mr.fei/pytorch-learn/data/images/dance.jpg")
    input_img = content_img.clone()
    output = transfer(content_img,style_img,input_img)
    imshow(output,'output_img')






