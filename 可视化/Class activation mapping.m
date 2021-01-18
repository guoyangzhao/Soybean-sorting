netName = "squeezenet";
% net = eval(netName);
% camera = webcam;

net = netTransfer;

inputSize = net.Layers(1).InputSize(1:2);
classes = net.Layers(end).Classes;
layerName = activationLayerName(netName);

% h = figure('Units','normalized','Position',[0.05 0.05 0.9 0.8],'Visible','on');

% while ishandle(h)
%     im = snapshot(camera);
%     im = imread('diseased_mid_2.jpg');
    im = imread('mask_bitten_mid_19.jpg');
    imResized = imresize(im,[inputSize(1), NaN]);
%     imResized  = im;
    imageActivations = activations(net,imResized,layerName);
    scores = squeeze(mean(imageActivations,[1 2]));
    
    if netName ~= "squeezenet"
        fcWeights = net.Layers(end-2).Weights;
        fcBias = net.Layers(end-2).Bias;
        scores =  fcWeights*scores + fcBias;
        
        [~,classIds] = maxk(scores,3);
        
        weightVector = shiftdim(fcWeights(classIds(1),:),-1);
        classActivationMap = sum(imageActivations.*weightVector,3);
    else
        [~,classIds] = maxk(scores,3);
        classActivationMap = imageActivations(:,:,classIds(1));
    end
    
    scores = exp(scores)/sum(exp(scores));     
    maxScores = scores(classIds);
    labels = classes(classIds);
    
%     subplot(1,2,1)
%     imshow(im)
    
%     CAMshow(im,classActivationMap)
%     title(string(labels) + ", " + string(maxScores));

figure
alpha = 0.1;

CAMshow(im, classActivationMap, alpha);
title(string(labels) + ", " + string(maxScores));

% plotGradCAM(im, classActivationMap, alpha);
% sgtitle(string(labels)+" (score: "+ string(maxScores)+")")

    drawnow
    
% end

% clear camera
% saveas(1,'cracked_xiao.jpg') %±£´æplot



% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% imSize = size(im);
% CAM = imresize(classActivationMap,imSize(1:2));
% CAM = normalizeImage(CAM);
% CAM(CAM<0.2) = 0;
% cmap = jet(255).*linspace(0,1,255)';
% CAM = ind2rgb(uint8(CAM*255),cmap)*255;
% 
% combinedImage = double(rgb2gray(im))/2 + CAM;
% combinedImage = normalizeImage(combinedImage)*255;
% % subplot(1,1,1); 
% imshow(uint8(combinedImage));
% aa=uint8(combinedImage);
% imwrite(aa,'mildew_mid_198_da3.jpg','jpg')


% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% normal
% cracked
% broken
% bitten
% diseased
% mildew





% % % % Function
function CAMshow(im,CAM,alpha)
imSize = size(im);
CAM = imresize(CAM,imSize(1:2));
CAM = normalizeImage(CAM);
CAM(CAM<0.2) = 0;
cmap = jet(255).*linspace(0,1,255)';
CAM = ind2rgb(uint8(CAM*255),cmap)*255;

combinedImage = double(rgb2gray(im))/2 + CAM;
combinedImage = normalizeImage(combinedImage)*255;

subplot(1,2,1); 
imshow(im); 
h = subplot(1,2,2);
imshow(uint8(combinedImage));


hold on;
imagesc(CAM,'AlphaData', alpha);
originalSize2 = get(h, 'Position');
colormap jet
colorbar
set(h, 'Position', originalSize2);

hold off;

end

function N = normalizeImage(I)
minimum = min(I(:));
maximum = max(I(:));
N = (I-minimum)/(maximum-minimum);
end

function layerName = activationLayerName(netName)

if netName == "squeezenet"
    layerName = 'relu_conv10';
elseif netName == "googlenet"
    layerName = 'inception_5b-output';
elseif netName == "resnet18"
    layerName = 'res5b_relu';
elseif netName == "mobilenetv2"
    layerName = 'out_relu';
%     layerName = 'Conv_1';
elseif netName == "shufflenet"
    layerName = 'node_199';
end

end








function plotGradCAM(img, gradcamMap, alpha)
% function plotGradCAM(img, alpha)
subplot(1,2,1)
imshow(img);

h= subplot(1,2,2);
imshow(img)
hold on;
imagesc(gradcamMap,'AlphaData', alpha);

originalSize2 = get(h, 'Position');

colormap jet
colorbar

set(h, 'Position', originalSize2);
hold off;
end