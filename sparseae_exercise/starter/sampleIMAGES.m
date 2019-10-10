function patches = sampleIMAGES()
% sampleIMAGES
% Returns 10000 patches for training
% 首先从IMAGES数据集中选择10张512*512图片，每张随机采样1000个大小为8*8的像素块，共得到10000个像素块，
% 对其重新排列（归一化）得到矩阵patches作为实验的样本数据集，patches为64*10000的矩阵

load IMAGES;    % load images from disk 
patchsize = 8;  % we'll use 8x8 patches 8*8小图块
numpatches = 10000; % 总共10000张
 
% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns.
patches = zeros(patchsize*patchsize, numpatches);
% 生成patchsize*patchsize行numpatches列的全零矩阵，即8*8行，10000列
%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data
%  from IMAGES. 
% 
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1
for imageNum  = 1:10 % 从每张图片中随机选择1000个patche，其中patch大小为8*8，共有10000个patch
    [rowNum colNum]= size(IMAGES(:,:,imageNum));
       for patchNum = 1:1000% 实现每张图片取1000个patch
           xPosition = randi([1,rowNum-patchsize+1]);%滑动平移，每次移动8*8
           yPosition = randi([1,colNum-patchsize+1]);
           % patches(:,(imageNum-1)*1000+patchNum)= ...即....表示续行
           patches(:,(imageNum-1)*1000+patchNum)=... 
           reshape(IMAGES(xPosition:xPosition+patchsize-1,yPosition:yPosition+patchsize-1,imageNum),64,1);
           % 对10张图像重新排序，第一张图像从第1列到第1000列，第二张图像从第1001列，直到第2000列，总共10000列
           % patches大小为64*10000
        end
end  
%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);
 
end
 
 
%% ---------------------------------------------------------------
function patches = normalizeData(patches)
 
% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer
 
% Remove DC (mean of images). 把patches数组中的每个元素值都减去mean(patches)
patches = bsxfun(@minus, patches, mean(patches));
% C=bsxfun(fun,A,B)表达的是两个数组A和B间元素的二值操作，fun是函数句柄或者m文件，或者是内嵌的函数。
% 在实际使用过程中fun有很多选择比如说加，减等，前面需要使用符号’@’.
% 一般情况下A和B需要尺寸大小相同，如果不相同的话，则只能有一个维度不同，同时A和B中在该维度处必须有一个的维度为1。
% 比如说bsxfun(@minus, A, mean(A))，其中A和mean(A)的大小是不同的，这里的意思需要先将mean(A)扩充到和A大小相同，然后用A的每个元素减去扩充后的mean(A)对应元素的值。
% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));%把patches的标准差变为其原来的3倍，即3gigma原则
patches = max(min(patches, pstd), -pstd) / pstd;
%  max(A,B)返回最大值，eg. A=[4 5 3;1 2 3] B=3 则将小于3的数换为3,即返回[4 5 3;3 3 3]
% sigma原则：数值分布在（μ-σ，μ+σ）中的概率为0.6526；
% 2sigma原则：数值分布在（μ-2σ，μ+2σ）中的概率为0.9544；
% 3sigma原则：数值分布在（μ-3σ，μ+3σ）中的概率为0.9974；
% 这里转换后将数据变到了-1到1之间，然后从-1到1转到0.1到0.9
% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;
 
end