clc
clear all
close all
%%
%数据管理
%1. 下载图像特征
img = imread('E:\Engineering issues during PhD\Utra2CT_imagedata\Complete ultrasound image data\Ultrasound image data\Clear data\Other organs data\cleanedimg\IM_0053\204.jpg');
%2. 下载参考数据
%我们暂时假设清晰数据的大小为0，因为这是理想状态
%事实上，我们采集大量样本之后，得到的结果是下面这种形式
%%
img=rgb2gray(img);%转换成灰度图
[k,l] = size(img);%计算图像的尺寸，后面需要构造一个跟这个差不多的矩阵来存放图像清晰度判断结果
[featurecell,featureVector,hogVisualization] = extractCell(double(img),'CellSize',[5 7],'BlockSize',[1 1],'NumBins',6);
figure(1)
imshow(img);
hold on;
% plot(hogVisualization);
% hold on;
%%

%怎么说呢，上面我们调用extractcell函数，将图像每个区域的cell特征计算出来
%接下来，我们首先根据上面计算出来的fraturecell特征
%首先，我们掌握一个什么思想呢，我们设置的cell是一个5*7的区域，我们将图像的梯度方向设定为6个方向。
%但是我们最后得到的梯度直方图是一个1*6的向量，因此我们首先需要计算好每个cell对应的直方图。
%然后，我们根据根据每个区域的cellhog判断该区域的图像质量，实现对图像质量评估的初始化。
%我们的初始化判断方法是使用该区域的特征和标准区域模糊或者清晰区域做标准差，然后根据标准差范围计算该区域的是模糊还是清晰。
%我们仍然通过滑动窗口的形式判断。
%1. 显然，我们期望的结果是，完全清晰，即图像的梯度该区域的梯度差为0，同时，他们的直方图特征为0，但是这种情况是很少的，一般出现这种情况，我们直接断该区域为清晰区域。
%因此，针对清晰区域和不清晰区域，我们需要建立相关的数据集来计算他们的平均值。
%%
%初始化函数
%首先计算一共有多少个窗口
%构造一个空矩阵，用于存放需要保存的
piex_init = zeros(size(img));
[numh,num_hang,num_lie] = size(featurecell);
%%
%初始化图像
init_image = init_function(img,featurecell,num_hang,num_lie);
%imwrite(init_image,'init_image.jpg');%保存图像
figure(2)
imshow(init_image);
hold on;
%%
%调用质量评估模型
%init_image(init_image>=1) = 0.9999;
%[num_piex1,Ising_qualt] = Isingmodel_mean(init_image);
Ising_qualt = Isingmodel(init_image);%进行质量评估，图像像素平滑处理
%num_piex1 = imrotate(num_piex1,-90);
Ising_qualt = imrotate(Ising_qualt,-90);%旋转
%Ising_qualt(Ising_qualt<0) = Ising_qualt(Ising_qualt<0)*(-1);
%imwrite(Ising_qualt,'Ising_qualt.jpg');%保存图像
figure(3)
imshow(Ising_qualt);
hold on

%%
%算法设计
%我们这次采用大窗口，每个窗口是16*16，在每个窗口内，我们诸葛计算像素的Ising能量模型。

%计算图像尺寸

%%
%这个函数时测验函数
function [num_piex1,Ising_img] = Isingmodel_mean(img)
%--------------------------------------------------------------------------
%%下面写一个基于Ising 能量模型的图像质量评估代码
%----------------------------------------
%该函数的主要思想是：这次，我们是实现像素级的质量评估，我们依然以窗口的形式处理图像
%我们的理论基础是，我们想办法让该窗口内的系统能量的到最小化。在图像质量清晰的地方，存在黑色斑块；在模糊区域存在模糊的斑块。如果让系统的能量达到最小，那么必然是需要让系统的内像素的状态统一
%最后，我们让清晰区域的黑色板块变成白色，让白色区域的黑色板块变白。
%需要指出的是，我们采用图像的余弦夹角作为图像质量评估指标，也就是说：
%当前像素梯度直方图与标准清晰像素直方图之间的夹角越小表示相似度越高，则此时余弦值越大。
%当前像素梯度直方图与标准清晰梯度直方图之间夹角越越大表示像素都越低，则此时余弦值越小
%余弦值在（0，1）之间，值越大，图像像素强度越亮。
%--------------------------------------------------------------------------
final_img = [];
num_piex = [];
img = imrotate(img,90);%旋转矩阵
[m,n] = size(img);

H =1;%外场磁场方向系数
%查看那个计算步骤导致心像素全部为NaN
for i = 9:m-9
    for j = 9:n-9
        if isnan(img(i,j))%如果当前像素值是NaN，那么则不处理该像素，默认为NaN。
            final_img(i,j) = img(i,j);
        else
            win_img = img(i-7:i+7,j-7:j+7);%提取以当前像素为中心的15*15的窗口内的像素。
            %ISing_avg = mean(win_img(:))+0.05;%平局值不行的话就得用
            ISing_avg = 0.65;%平局值不行的话就得用
            Num = size(win_img,1)*size(win_img,2);%计算窗口内像素中总数。
            Numlowavg = sum(sum(win_img<ISing_avg));%比平均值小的都是黑色
            Numhigavg = sum(sum(win_img>ISing_avg));%比平均值大的就是白色，这个是计算白色像素的个数。
            %如果numhigavg的值大于15*15的一半，那么这个区域可以判定为清晰区域，但是此时要把小于平均值的地方都给换成负数，从而改变状态
            %win_img(win_img<0) = ISing_avg*(-1);%最后判断矩阵中是否还有负的像素值
            %win_img = abs(win_img);
            final_img(i,j) = mean(win_img(:))+1.2*exp(-i-10);
            num_piex(i,j) = Numhigavg;
        end
    end
end
Ising_img = final_img;
num_piex1 = num_piex;
end
%%
%这个函数是求Ising model的函数。
function Ising_img = Isingmodel(img)
%--------------------------------------------------------------------------
%%下面写一个基于Ising 能量模型的图像质量评估代码
%----------------------------------------
%该函数的主要思想是：这次，我们是实现像素级的质量评估，我们依然以窗口的形式处理图像
%我们的理论基础是，我们想办法让该窗口内的系统能量的到最小化。在图像质量清晰的地方，存在黑色斑块；在模糊区域存在模糊的斑块。如果让系统的能量达到最小，那么必然是需要让系统的内像素的状态统一
%最后，我们让清晰区域的黑色板块变成白色，让白色区域的黑色板块变白。
%需要指出的是，我们采用图像的余弦夹角作为图像质量评估指标，也就是说：
%当前像素梯度直方图与标准清晰像素直方图之间的夹角越小表示相似度越高，则此时余弦值越大。
%当前像素梯度直方图与标准清晰梯度直方图之间夹角越越大表示像素都越低，则此时余弦值越小
%余弦值在（0，1）之间，值越大，图像像素强度越亮。
%--------------------------------------------------------------------------
final_img = [];
img = imrotate(img,90);%旋转矩阵
[m,n] = size(img);
H =1;%外场磁场方向系数
%查看那个计算步骤导致心像素全部为NaN
for i = 9:m-9
    for j = 9:n-9
        if isnan(img(i,j))%如果当前像素值是NaN，那么则不处理该像素，默认为NaN。
            final_img(i,j) = img(i,j);
        else
            win_img = img(i-7:i+7,j-7:j+7);%提取以当前像素为中心的15*15的窗口内的像素。
            ISing_avg1 = mean(win_img(:));%计算该窗口内的平均值，或者设计为
            ISing_avg = 0.6;%平局值不行的话就得用
            Num = size(win_img,1)*size(win_img,2);%计算窗口内像素中总数。
            %我们的期望是，在近场区域Numlowavg的值小于NumHalf，因为一般来讲，这个区域的像素是清晰的。
            %在远场区域，Numlowavg的值大于NumHalf，这个区域与一般是模糊区域。
            NumHalf = ceil((size(win_img,1)*size(win_img,1))/2);
            Numlowavg = sum(sum(win_img <= ISing_avg));%比平均值小的是黑色，
            Numhigavg = sum(sum(win_img > ISing_avg)); %比平均值大的就是白色，这个是计算白色像素的个数。
            win_img(win_img<ISing_avg) = ISing_avg*(-1);%当图像中像素值小于平局值时，则判定该像素状态为模糊状态，即“down”，因此取值为负
            %E = win_img(ceil(size(win_img,1)./2),ceil(size(win_img,2)./2));%提取当前像素
            %下面是判断窗口中心像素的符号，这边判断的时候，应该用平均值判断，还是用0.5判断？需要试验一下
            %自动判断中心像素颜色,中心像素就是当前像素。
            if (Numlowavg <= NumHalf && win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))<0)%如果在白色区域，中心像素是黑色的，那么就把它转换过来
                img(i,j) = win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))*(-1);
            elseif (Numlowavg > NumHalf && win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))>0)%如果在黑色区域中心像素是白色的，那么也要把它转换过来
                img(i,j) = win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))*(-1);
            else%其他情况下像素变
                img(i,j) = win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5));
            end
            M = img(i,j);%提取当前像素的像素值
            %计算win内的能量值，并根据delta_E更新像素。
            for wi = 1:size(win_img,1)
                for wj = 1:size(win_img,2)
                    E_init = (-1)*sum(sum(M.*win_img)) - H*(Numhigavg-Num*0.5).*M;%没有调整像素值时的能量
                    %首先判断是在清晰区域还是在模糊区域
                    if Numlowavg > NumHalf %黑色少表示在清晰区域，在清晰区域，我们强制默认中心像素的像素状态为正，如果不是正的，我们就要强制让其为正。
                        if win_img(wi,wj) < 0%如果在清晰区域，开始逐个判断像素属性，如果是负数，则将其转换为平均值
                            win_img(wi,wj) = ISing_avg1;
                            E = (-1)*sum(sum(M.*win_img)) - H*(Numhigavg-Num*0.5).*M;%调整像素值后的能量值。
                            delta_E = E-E_init;%计算调整后能量的变化，我们的期望是，调整后系统的能量会逐渐变低的，所以delta_E会小于零
                            if delta_E < 0
                                win_img(wi,wj) = ISing_avg1 + exp(-i-10) ;%
                            else
                                win_img(wi,wj) = ISing_avg1  ;%.*rand(1).*0.9;
                            end
                        else
                            win_img(wi,wj) = win_img(wi,wj) ;
                        end
                    else     %否则就是在模糊区域，在模糊区域，我们期望将白色的改成黑色的。
                        if win_img(wi,wj) > 0
                            win_img(wi,wj) = (ISing_avg1)*(-1);
                            E = (-1)*sum(sum(M.*win_img)) - H*(Numhigavg-Num*0.5).*M;%调整像素值后的能量值
                            delta_E = E-E_init;%计算调整后能量的变化，我们的期望是，调整后系统的能量会逐渐变低的，所以delta_E会小于零
                            if delta_E < 0
                                win_img(wi,wj) = (ISing_avg1)-exp(-i-10);
                            else
                                win_img(wi,wj) = (ISing_avg1);%
                            end
                        else
                            win_img(wi,wj) = win_img(wi,wj);
                        end
                        if win_img(wi,wj) < 0
                            win_img(wi,wj) = win_img(wi,wj)*(-1);
                        else
                            win_img(wi,wj) = win_img(wi,wj) ;
                        end
                    end
                end
            end
            %win_img(win_img<0) = ISing_avg*(-1);%最后判断矩阵中是否还有负的像素值
            %win_img = abs(win_img);
            final_img(i,j) = mean(win_img(:));
        end
    end
end
Ising_img = final_img;
end

%%
%首先构造循环，实现对每个cell的遍历
function init_image = init_function(img,featurecell,num_hang,num_lie)
%---------------------------------------------------------------------------
%模糊的情况下如何赋值.初步计算，我们可以通过计算两个向量之间的余弦值作为像素的权值。
%我们的依据是两个向量夹角余弦越大表示两个向量的夹角越小，夹角余弦越小表示两向量的夹角越大。
%当两个向量的方向重合时夹角余弦取最大值1，当两个向量的方向完全相反夹角余弦取最小值-1。
%在该部分，我们取夹角余弦的绝对值，作为评价标准
%我们的目的是让越清晰的图像区域越白，相反，越模糊的图像区域越黑
%————————————————————————————————————————
r = 1;%行1，2，3，4，....16
c = 1;%列1，2，3，4，....16
img_fiff = diff_sun(img);
if img_fiff >1
    qxfeature = [0.353    0.00190    0.00908    0.00550    0.0320    0.527];%针对模糊的
else
    qxfeature = [0.0353    0.190    0.00908    0.00550    0.320    0.0527];
end
for i = 1:num_hang%;
    for j = 1:num_lie%
        %下面开始从featurecell中提取每个cell的hog特征
        hogi = featurecell(:,i,j);%i每循环一次，算法会提取一行的cell特征
        hogi = reshape(hogi,1,6);
        %后期使用训练好的SVM模型作为初始化标准，在目前的试验中，我们使用标准欧几里得距离来计算
        D = pdist([hogi;qxfeature],'euclidean');%标准化欧氏距离 (Standardized Euclidean distance)
        C = 1-abs(pdist([qxfeature;hogi], 'cosine'));%目标之间的余弦夹角
        %通过if条件句判断该cell所包含区域的像素属于清晰还是模糊
        %这边的主要盘踞还是夹角余弦
        if img_fiff >1
            piex_init(r:r+4,c:c+6) =1*(1./(1./(1+exp(-i*0.1+num_hang./15))+0.8))*((C*1.1+D*0.35)/1.8);%针对模糊的
        else
            piex_init(r:r+4,c:c+6) =1*(1./(1./(1+exp(-i*0.1+num_hang./15))+0.8))*((C*1.6+D*0.35)/1.5);%目前比较满意状态
        end
        %piex_init(r:r+4,c:c+6) =1*(1./(1./(1+exp(-i*0.1+num_hang./15))+0.8))*((C*1.6+D*0.35)/1.5);%目前比较满意状态
        %piex_init(r:r+4,c:c+6) =1*(1./(1./(1+exp(-i*0.1+num_hang./15))+0.8))*((C*1.1+D*0.35)/1.2);%目前比较满意状态
        %piex_init(r:r+4,c:c+6) =1*(1./(1./(1+exp(-i*0.1+num_hang./15))+0.8))*((C*1.2+D*0.65)/1.6);%目前比较满意状态
        %piex_init(r:r+4,c:c+6) =1*(1./(1./(1+exp(-i*0.1+num_hang./15))+0.8))*((1-pdist([qxfeature;hogi], 'cosine')+D*0.75)/1.6);%目前比较满意状态
        
        c = c + 7;
    end
    r = r+5;
    c = 1;%每次列内循环结束之后要把列数归一，这样方便从第二行开始继续从1开始。
end
init_image = piex_init;
end

%%
%矩阵归一化函数
function [ Result ] = normalize( Data,lowbound,upbound )
%将数据矩阵Data进行规范化
%新的上界是upbound,新的下界是lowbound
%要求输入的数据矩阵中每一行表示一个多维样本点

msize = size(Data);
Result = zeros(msize(1),msize(2));%存储结果
mins = Data(1,:);%储存每一维的最小值
maxs = Data(1,:);%储存每一维的最大值

%%%%
% 计算每一维度的最小值与最大值
%
%%%%
for i = 1:msize(1)
    for j = 1:msize(2)
        if Data(i,j)maxs(j)
            maxs(j) = Data(i,j);
        end
    end
end

for i = 1:msize(1)
    for j = 1:msize(2)
        Result(i,j) = lowbound + (Data(i,j)-mins(j))/(maxs(j)-mins(j))*(upbound-lowbound);
    end
end

end
%%
%图像降噪函数
function img_fiff = diff_sun(img)

[m,n] = size(img);
Y=dct2(img);
I=zeros(m,n);
%高频屏蔽
I(1:m/3,1:n/3)=1;
Ydct=Y.*I;
%逆DCT变换
img_filter=uint8(idct2(Ydct));
img_fiff1 = img_filter - img;
img_fiff = mean(img_fiff1(:));
end

%%
%FHOGIM函数
function Isingvalue = Ising_grand(img)
img=rgb2gray(img);%变成灰度图
img = single(img);%单精度，提高运行效率
[Fx,Fy]= gradient(img);
q = 13;
Fx(Fx>-q & Fx<0) = 0;
Fx(Fx<q & Fx>0) = 0;
[m,n] = size(Fx);
E_r = [];
for i = 2:m-1
    for j = 2:n-1
        E_r(i,j) = ((Fx(i,j)*Fx(i,j-1)+ Fx(i,j)*Fx(i-1,j) + Fx(i,j)*Fx(i,j+1)...
            + Fx(i,j)*Fx(i+1,j))-Fx(i,j));%求图像中每个像素的易辛能量值。
    end
end
Isingvalue = mean(E_r(:));
y1 = 3.4*log(Isingvalue-3.1);
Isingvalue = (1.1)./(1+exp(-y1+10.2));
end
