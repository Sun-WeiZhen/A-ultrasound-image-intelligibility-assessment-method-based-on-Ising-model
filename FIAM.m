clc
clear all
close all
%%
%计算每个图像的噪声率
%这边是心脏部分的，源自心脏专用设备
% str='E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0050\'; %待处理数据的保存路径；28里面有好用的，41里有好用的
% %计算文件中图片的数量
% pics=dir('E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0050\*.jpg');
%这边是其他部分的，源自另一个机器
str='E:\Engineering issues during PhD\Utra2CT_imagedata\Complete ultrasound image data\Ultrasound image data\Clear data\Other organs data\cleanedimg\IM_0009\'; %待处理数据的保存路径；28里面有好用的，41里有好用的
%计算文件中图片的数量
pics=dir('E:\Engineering issues during PhD\Utra2CT_imagedata\Complete ultrasound image data\Ultrasound image data\Clear data\Other organs data\cleanedimg\IM_0009\*.jpg');
D=length(pics);
diffvalue = zeros(1,D);
tic%开启时间统计
for i=1:D
    img=imread([str,num2str(i),'.jpg']); %依次读取每一幅图像
    Isingvalue = Ising_grand(img);
    diffvalue(1,i) = Isingvalue;
end
toc%关闭时间统计
diffvalue;
figure(1)
i=1:D;
plot(i,diffvalue,'k');
hold on
%%
%函数名
%Ising_grand   isingmodel
%imageentropy  图像熵
%imagecontrast 对比度
%EnergyofGradient 梯度能量函数EOG
%Tenengrad1     Tenengrad函数
%Brenner1       Brenner函数


%%

% final_img = [];
% img = imrotate(single(img1),90);%旋转矩阵
% [m,n] = size(img);
% H =1;%外场磁场方向系数
%查看那个计算步骤导致心像素全部为NaN
% for i = 9:m-9
%     for j = 9:n-9
%         if isnan(img(i,j))%如果当前像素值是NaN，那么则不处理该像素，默认为NaN。
%             final_img(i,j) = img(i,j);
%         else
%             win_img = img(i-7:i+7,j-7:j+7);%提取以当前像素为中心的15*15的窗口内的像素。
%             %计算win内的能量值，并根据delta_E更新像素。
% %             for wi = 1:size(win_img,1)
% %                 for wj = 1:size(win_img,2)
% %
% %                 end
% %             end
%             as = mean(win_img(:));
%             %win_img(win_img<0) = ISing_avg*(-1);%最后判断矩阵中是否还有负的像素值
%             %win_img = abs(win_img);
%             final_img(i,j) = as;
%         end
%     end
% end
% Ising_img = final_img;
% Ising_img = imrotate(Ising_img,-90);%旋转矩阵
% imshow(Ising_img);
% improfile
%%
function Ising_img = Isingmodel_mean(img)
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
            
            %win_img(win_img<0) = ISing_avg*(-1);%最后判断矩阵中是否还有负的像素值
            %win_img = abs(win_img);
            final_img(i,j) = mean(win_img(:));
        end
    end
end
Ising_img = final_img;
end


%%
%备份函数

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
            ISing_avg = mean(win_img(:))+0.25;%平局值不行的话就得用
            Num = size(win_img,1)*size(win_img,2);%计算窗口内像素中总数。
            Numlowavg = sum(sum(win_img>ISing_avg));%比平均值大的就是白色，这个是计算白色像素的个数。
            Numhigavg = sum(sum(win_img<ISing_avg));
            win_img(win_img<ISing_avg) = ISing_avg*(-1);%当图像中像素值小于平局值时，则判定该像素状态为模糊状态，即“down”，因此取值为负
            %E = win_img(ceil(size(win_img,1)./2),ceil(size(win_img,2)./2));%提取当前像素
            %下面是判断窗口中心像素的符号，这边判断的时候，应该用平均值判断，还是用0.5判断？需要试验一下
            %自动判断中心像素颜色,中心像素就是当前像素。
            if (Numlowavg <= Numhigavg && win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))<0)%如果在白色区域，中心像素是黑色的，那么就把它转换过来
                img(i,j) = win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))*(-1);
            elseif (Numlowavg > Numhigavg && win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))>0)%如果在黑色区域中心像素是白色的，那么也要把它转换过来
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
                    if Numlowavg <= Numhigavg %黑色少表示在清晰区域，在清晰区域，我们强制默认中心像素的像素状态为正，如果不是正的，我们就要强制让其为正。
                        if win_img(wi,wj)<0%如果在清晰区域，开始逐个判断像素属性，如果是负数，则将其转换为平均值
                            win_img(wi,wj) = ISing_avg;
                            E = (-1)*sum(sum(M.*win_img)) - H*(Numhigavg-Num*0.5).*M;%调整像素值后的能量值。
                            delta_E = E-E_init;%计算调整后能量的变化，我们的期望是，调整后系统的能量会逐渐变低的，所以delta_E会小于零
                            if delta_E < 0
                                win_img(wi,wj) = ISing_avg - exp(-i-10) ;%
                            else
                                win_img(wi,wj) = ISing_avg - exp(-i-10) ;%.*rand(1).*0.9;
                            end
                        else
                            win_img(wi,wj) = win_img(wi,wj)- exp(-i-10) ;
                        end
                    else     %否则就是在模糊区域，在模糊区域，我们期望将白色的改成黑色的。
                        if win_img(wi,wj) > 0
                            win_img(wi,wj) = (ISing_avg-0.3)*(-1);
                            E = (-1)*sum(sum(M.*win_img)) - H*(Numhigavg-Num*0.5).*M;%调整像素值后的能量值
                            delta_E = E-E_init;%计算调整后能量的变化，我们的期望是，调整后系统的能量会逐渐变低的，所以delta_E会小于零
                            if delta_E < 0
                                win_img(wi,wj) = (ISing_avg-0.3).*exp(-i-20);
                            else
                                win_img(wi,wj) = (ISing_avg-0.3).*exp(-i-20);%
                            end
                        else
                            win_img(wi,wj) = win_img(wi,wj).*exp(-i-20);
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

function img_filter = filter_sun(img)
img=rgb2gray(img);
[m,n] = size(img);
Y=dct2(img);
I=zeros(m,n);
%高频屏蔽
I(1:m/3,1:n/3)=1;
Ydct=Y.*I;
%逆DCT变换
img_filter=uint8(idct2(Ydct));
end

function [PSNR] = eval_psnr(img,imgn)
% =================PSNR评价
% param ：
%       img:输入灰度图像（img与imgn同等大小）
%       imgn:输入要进行对比的灰度图像
%
B = 8;                %编码一个像素用多少二进制位
MAX = 2^B-1;          %图像有多少灰度级
[height,width,~] = size(img);
MES = sum(sum((img-imgn).^2))/(height*width);     %均方差
PSNR = 20*log10(MAX/sqrt(MES));           %峰值信噪比

end

%%
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

%%
%图像熵
function imgentr = imageentropy(img)
img=rgb2gray(img);%变成灰度图
img = single(img);%单精度，提高运行效率
[C,L]=size(img); %求图像的规格
Img_size=C*L; %图像像素点的总个数
G=256; %图像的灰度级
H_x=0;
nk=zeros(G,1);%产生一个G行1列的全零矩阵
for i=1:C
    for j=1:L
        Img_level=img(i,j)+1; %获取图像的灰度级
        nk(Img_level)=nk(Img_level)+1; %统计每个灰度级像素的点数
    end
end
for k=1:G  %循环
    Ps(k)=nk(k)/Img_size; %计算每一个像素点的概率
    if Ps(k)~=0; %如果像素点的概率不为零
        H_x=-Ps(k)*log2(Ps(k))+H_x; %求熵值的公式
    end
end
imgentr = H_x;  %显示熵值
end

%%
%对比度
function imgcontrast = imagecontrast(img)
img=rgb2gray(img);%变成灰度图
img = single(img);%单精度，提高运行效率
img = single(img);
[m n] = size(img);
k = 0;
d = mean(mean(img));
for i = 1:m
    for j = 1:n
        k = k+(img(i,j)-d)^2;
    end
end
imgcontrast = sqrt(k/m*n);
end

%%
%Energy of Gradient
function EOG = EnergyofGradient(img)
img=rgb2gray(img);%变成灰度图
img = single(img);%单精度，提高运行效率
[M N]=size(img);
FI=0;
for x=1:M-1
    for y=1:N-1
        % x方向和y方向的相邻像素灰度值只差的的平方和作为清晰度值
        FI=FI+(img(x+1,y)-img(x,y))*(img(x+1,y)-img(x,y))+(img(x,y+1)-img(x,y))*(img(x,y+1)-img(x,y));
    end
end
EOG = FI;
end

%%
%Tenengradhanshu
function Tenengrad = Tenengrad1(img)
img=rgb2gray(img);%变成灰度图
img = single(img);%单精度，提高运行效率
[M N]=size(img);
%利用sobel算子gx,gy与图像做卷积，提取图像水平方向和垂直方向的梯度值
GX = 0;   %图像水平方向梯度值
GY = 0;   %图像垂直方向梯度值
FI = 0;   %变量，暂时存储图像清晰度值
T  = 0;   %设置的阈值
for x=2:M-1
    for y=2:N-1
        GX = img(x-1,y+1)+2*img(x,y+1)+img(x+1,y+1)-img(x-1,y-1)-2*img(x,y-1)-img(x+1,y-1);
        GY = img(x+1,y-1)+2*img(x+1,y)+img(x+1,y+1)-img(x-1,y-1)-2*img(x-1,y)-img(x-1,y+1);
        SXY= sqrt(GX*GX+GY*GY); %某一点的梯度值
        %某一像素点梯度值大于设定的阈值，将该像素点考虑，消除噪声影响
        if SXY>T
            FI = FI + SXY*SXY;    %Tenengrad值定义
        end
    end
end
Tenengrad = FI;
end

%%
%Brenner函数
function Brenner = Brenner1(img)
img=rgb2gray(img);%变成灰度图
img = single(img);%单精度，提高运行效率
[M N]=size(img);     %M等于矩阵行数，N等于矩阵列数；size()获取矩阵行列
FI=0;        %变量，暂时存储每一幅图像的Brenner值
for x=1:M-2      %Brenner函数原理，计算相差两个位置的像素点的灰度值
    for y=1:N
        FI=FI+(img(x+2,y)-img(x,y))*(img(x+2,y)-img(x,y));
    end
end
Brenner = FI;
end













