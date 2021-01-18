%加载预训练网络
net = squeezenet;

%阅读并显示图像。保存其大小以备将来使用。
im = imread('face.jpg');
imshow(im)
imgSize = size(im);
imgSize = imgSize(1:2);

%分析网络以查看可以查看的图层。
analyzeNetwork(net)

%显示第一个卷积层的激活
act1 = activations(net,im,'conv1');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[8 8]);
imshow(I)

%调整通道22中的激活大小，使其具有与原始图像相同的大小，并显示激活。
act1ch22 = act1(:,:,:,22);
act1ch22 = mat2gray(act1ch22);
act1ch22 = imresize(act1ch22,imgSize);
I = imtile({im,act1ch22});
imshow(I)

%寻找最强的激活渠道
[maxValue,maxValueIndex] = max(max(max(act1)));
act1chMax = act1(:,:,:,maxValueIndex);
act1chMax = mat2gray(act1chMax);
act1chMax = imresize(act1chMax,imgSize);
I = imtile({im,act1chMax});
imshow(I)


