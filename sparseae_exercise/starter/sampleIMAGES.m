function patches = sampleIMAGES()
% sampleIMAGES
% Returns 10000 patches for training
% ���ȴ�IMAGES���ݼ���ѡ��10��512*512ͼƬ��ÿ���������1000����СΪ8*8�����ؿ飬���õ�10000�����ؿ飬
% �����������У���һ�����õ�����patches��Ϊʵ����������ݼ���patchesΪ64*10000�ľ���

load IMAGES;    % load images from disk 
patchsize = 8;  % we'll use 8x8 patches 8*8Сͼ��
numpatches = 10000; % �ܹ�10000��
 
% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns.
patches = zeros(patchsize*patchsize, numpatches);
% ����patchsize*patchsize��numpatches�е�ȫ����󣬼�8*8�У�10000��
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
for imageNum  = 1:10 % ��ÿ��ͼƬ�����ѡ��1000��patche������patch��СΪ8*8������10000��patch
    [rowNum colNum]= size(IMAGES(:,:,imageNum));
       for patchNum = 1:1000% ʵ��ÿ��ͼƬȡ1000��patch
           xPosition = randi([1,rowNum-patchsize+1]);%����ƽ�ƣ�ÿ���ƶ�8*8
           yPosition = randi([1,colNum-patchsize+1]);
           % patches(:,(imageNum-1)*1000+patchNum)= ...��....��ʾ����
           patches(:,(imageNum-1)*1000+patchNum)=... 
           reshape(IMAGES(xPosition:xPosition+patchsize-1,yPosition:yPosition+patchsize-1,imageNum),64,1);
           % ��10��ͼ���������򣬵�һ��ͼ��ӵ�1�е���1000�У��ڶ���ͼ��ӵ�1001�У�ֱ����2000�У��ܹ�10000��
           % patches��СΪ64*10000
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
 
% Remove DC (mean of images). ��patches�����е�ÿ��Ԫ��ֵ����ȥmean(patches)
patches = bsxfun(@minus, patches, mean(patches));
% C=bsxfun(fun,A,B)��������������A��B��Ԫ�صĶ�ֵ������fun�Ǻ����������m�ļ�����������Ƕ�ĺ�����
% ��ʵ��ʹ�ù�����fun�кܶ�ѡ�����˵�ӣ����ȣ�ǰ����Ҫʹ�÷��š�@��.
% һ�������A��B��Ҫ�ߴ��С��ͬ���������ͬ�Ļ�����ֻ����һ��ά�Ȳ�ͬ��ͬʱA��B���ڸ�ά�ȴ�������һ����ά��Ϊ1��
% ����˵bsxfun(@minus, A, mean(A))������A��mean(A)�Ĵ�С�ǲ�ͬ�ģ��������˼��Ҫ�Ƚ�mean(A)���䵽��A��С��ͬ��Ȼ����A��ÿ��Ԫ�ؼ�ȥ������mean(A)��ӦԪ�ص�ֵ��
% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));%��patches�ı�׼���Ϊ��ԭ����3������3gigmaԭ��
patches = max(min(patches, pstd), -pstd) / pstd;
%  max(A,B)�������ֵ��eg. A=[4 5 3;1 2 3] B=3 ��С��3������Ϊ3,������[4 5 3;3 3 3]
% sigmaԭ����ֵ�ֲ��ڣ���-�ң���+�ң��еĸ���Ϊ0.6526��
% 2sigmaԭ����ֵ�ֲ��ڣ���-2�ң���+2�ң��еĸ���Ϊ0.9544��
% 3sigmaԭ����ֵ�ֲ��ڣ���-3�ң���+3�ң��еĸ���Ϊ0.9974��
% ����ת�������ݱ䵽��-1��1֮�䣬Ȼ���-1��1ת��0.1��0.9
% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;
 
end