

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imds = imageDatastore('224机器数据集_520', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
[XTrain,XValidation] = splitEachLabel(imds,0.7,'randomized');
net = netTransfer;

% % % Classify Validation Data
figure();
YPred = classify(net,XValidation);
confusionchart(XValidation.Labels,YPred,'ColumnSummary',"column-normalized")

% % % Compute Activations for Several Layers %第一层池化 最后一层卷积 softmax
% global_average_pooling2d_1
% Logits
% Logits_softmax

% earlyLayerName = "global_average_pooling2d_1";
% finalConvLayerName = "new_fc";
% softmaxLayerName = "Logits_softmax";
earlyLayerName = "efficientnet-b0|model|blocks_0|se|GlobAvgPool";
finalConvLayerName = "efficientnet-b0|model|head|conv2d|Conv2D";
softmaxLayerName = "Softmax";
pool1Activations = activations(net,...
    XValidation,earlyLayerName,"OutputAs","rows");
finalConvActivations = activations(net,...
    XValidation,finalConvLayerName,"OutputAs","rows");
softmaxActivations = activations(net,...
    XValidation,softmaxLayerName,"OutputAs","rows");

% % % Ambiguity of Classifications
[R,RI] = maxk(softmaxActivations,2,2);
ambiguity = R(:,2)./R(:,1);

[ambiguity,ambiguityIdx] = sort(ambiguity,"descend");

classList = unique(XValidation.Labels);
top10Idx = ambiguityIdx(1:10);
top10Ambiguity = ambiguity(1:10);
mostLikely = classList(RI(ambiguityIdx,1));
secondLikely = classList(RI(ambiguityIdx,2));
table(top10Idx,top10Ambiguity,mostLikely(1:10),secondLikely(1:10),XValidation.Labels(ambiguityIdx(1:10)),...
    'VariableNames',["Image #","Ambiguity","Likeliest","Second","True Class"])

v = 27;
figure();
imshow(XValidation.Files{v});
title(sprintf("Observation: %i\n" + ...
    "Actual: %s. Predicted: %s", v, ...
    string(XValidation.Labels(v)), string(YPred(v))), ...
    'Interpreter', 'none');

% % % Compute 2-D Representations of Data Using t-SNE
rng default
pool1tsne = tsne(pool1Activations);
finalConvtsne = tsne(finalConvActivations);
softmaxtsne = tsne(softmaxActivations);

% % % Compare Network Behavior for Early and Later Layers
doLegend = 'off';
Legend = 'on';
markerSize = 7;
figure;

subplot(1,3,1);
gscatter(pool1tsne(:,1),pool1tsne(:,2),XValidation.Labels, ...
    [],'.',markerSize,Legend);
title("Max pooling activations");

subplot(1,3,2);
gscatter(finalConvtsne(:,1),finalConvtsne(:,2),XValidation.Labels, ...
    [],'.',markerSize,doLegend);
title("Final conv activations");

subplot(1,3,3);
gscatter(softmaxtsne(:,1),softmaxtsne(:,2),XValidation.Labels, ...
    [],'.',markerSize,doLegend);
title("Softmax activations");

% l = legend;%%%
% l.Interpreter = "none";%%%
% l.Location = "bestoutside";%%%

% % % Explore Observations in t-SNE Plot
numClasses = length(classList);
colors = lines(numClasses);
h = figure;
gscatter(softmaxtsne(:,1),softmaxtsne(:,2),XValidation.Labels,colors);
l = legend;
l.Interpreter = "none";
l.Location = "bestoutside";

% obs = 99;
% figure(h)
% 
% hold on;
% hs = scatter(softmaxtsne(obs, 1), softmaxtsne(obs, 2), ...
%     'black','LineWidth',1.5);
% l.String{end} = 'Hamburger';
% hold off;
% figure();
% imshow(XValidation.Files{obs});
% title(sprintf("Observation: %i\n" + ...
%     "Actual: %s. Predicted: %s", obs, ...
%     string(XValidation.Labels(obs)), string(YPred(obs))), ...
%     'Interpreter', 'none');
% 
% obs = 27;
% figure(h)
% hold on;
% h = scatter(softmaxtsne(obs, 1), softmaxtsne(obs, 2), ...
%     'k','d','LineWidth',1.5);
% l.String{end} = 'French Fries';
% hold off;

% % % Helper Function
% function downloadExampleFoodImagesData(url, dataDir)
% % Download the Example Food Image data set, containing 978 images of
% % different types of food split into 9 classes.
% 
% % Copyright 2019 The MathWorks, Inc.
% 
% fileName = "ExampleFoodImageDataset.zip";
% fileFullPath = fullfile(dataDir, fileName);
% 
% % Download the .zip file into a temporary directory.
% if ~exist(fileFullPath, "file")
%     fprintf("Downloading MathWorks Example Food Image dataset...\n");
%     fprintf("This can take several minutes to download...\n");
%     websave(fileFullPath, url);
%     fprintf("Download finished...\n");
% else
%     fprintf("Skipping download, file already exists...\n");
% end
% 
