# Neural_style_tansfer
基于pytorch 框架进行神经风格转移

Content Reconstructions：我们可以通过网络中某一层的feature map来重建输入图像，从而可以可视化CNN不同层中图像的处理信息。这相当于一个逆向的过程，输入的是特征，输出的是图像。底层的细节保留完整（颜色、纹理信息），高层则保留一些高层次的语义信息，如整体的内容和空间信息会保存等。


Style Reconstructions ：我们计算不同层中的不同feature之间的关系来表示风格。我们通过风格表征来进行风格重建。随着层数的累加，纹理越来越完整，平滑。为了可视化网络中不同层次的图像信息，我们输入一张初始化的白噪音图片，使它尽可能的去匹配原始图片在网络特定层的feature maps，我们利用梯度下降，不断缩小原始图片与生成图片之间的feature的差异。保持网络的权重不变，只不过是不断更新生成图片的像素值，从而实现图像重建。


对于内容损失，通常使用均方误差函数。



对于风格损失，我们通过计算不同feature maps之间相关性来表示图片的纹理信息。我们可以使用Gram matrix来计算这种相关性。Gram Matrix实际上可看做是feature之间的偏心协方差矩阵（即没有减去均值的协方差矩阵），在feature map中，每一个数字都来自于一个特定滤波器在特定位置的卷积，因此每个数字就代表一个特征的强度，而Gram计算的实际上是两两特征之间的相关性，当同一个维度上面的值相乘的时候原来越小就变得更小，原来越大就变得越大；二个不同维度上的关系也在相乘的表达当中表示出来。哪两个特征是同时出现的，哪两个是此消彼长的等等，同时，Gram的对角线元素，还体现了每个特征在图像中出现的量，因此，Gram矩阵有助于把握整个图像的大体风格。有了表示风格的Gram Matrix，要度量两个图像风格的差异，只需比较他们Gram Matrix的差异即可。这就将一个抽象的问题具体化了。通常使用Gram Matrix 的均方误差来代表风格损失。
