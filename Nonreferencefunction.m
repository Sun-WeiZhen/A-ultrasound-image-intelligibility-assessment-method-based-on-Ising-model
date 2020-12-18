clc
clear all
close all

%%
%图像熵----质量评估方法
str='E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0041\'; %待处理数据的保存路径；28里面有好用的，41里有好用的
%计算文件中图片的数量
pics=dir('E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0041\*.jpg');
D=length(pics);
diffvalue = zeros(1,D);
tic%开启时间统计
for t=1:D
    I=imread([str,num2str(t),'.jpg']); %依次读取每一幅图像
    I=single(I);
    I=rgb2gray(I);%变成灰度图
    [C,L]=size(I); %求图像的尺寸
    Img_size=C*L; %图像像素点的总个数
    G=256; %图像的灰度级
    H_x=0;
    nk=zeros(G,1);%产生一个G行1列的全零矩阵
    for i=1:C
        for j=1:L
            Img_level=I(i,j)+1; %获取图像的灰度级
            nk(Img_level)=nk(Img_level)+1; %统计每个灰度级像素的点数
        end
    end
    for k=1:G  %循环
        Ps(k)=nk(k)/Img_size; %计算每一个像素点的概率
        if Ps(k)~=0 %如果像素点的概率不为零
            H_x=-Ps(k)*log2(Ps(k))+H_x; %求熵值的公式
        end
    end
    diffvalue(1,t) = H_x;
end
toc%关闭时间统计
figure(1)
plot(diffvalue);
hold on

%%
%基于Isingmodel的方法.方法的优点，算法对于图像质量的变化更为敏感
%计算每个图像的噪声率
% str='E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0050\'; %待处理数据的保存路径；28里面有好用的，41里有好用的
% %计算文件中图片的数量
% pics=dir('E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0050\*.jpg');
% D=length(pics);
% diffvalue = zeros(1,D);
% tic%开启时间统计
% for i=1:D
%     img=imread([str,num2str(i),'.jpg']); %依次读取每一幅图像
%     Isingvalue = Ising_grand(img);
%     diffvalue(1,i) = Isingvalue;
% end
% toc%关闭时间统计
% figure(1)
% plot(diffvalue);
% hold on

%%
%计算对比度

%      str='E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0050\'; %待处理数据的保存路径；28里面有好用的，41里有好用的
% %计算文件中图片的数量
% pics=dir('E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0050\*.jpg');
% D=length(pics);
% diffvalue = zeros(1,D);
% tic%开启时间统计
% for t=1:D
%     img=imread([str,num2str(t),'.jpg']); %依次读取每一幅图像
%     value = image_contrast(img);
%     diffvalue(1,t) = value;
% end
% toc%关闭时间统计
% figure(1)
% plot(diffvalue);
% hold on

%%
%能量梯度函数value = EOG(img)
%      str='E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0050\'; %待处理数据的保存路径；28里面有好用的，41里有好用的
% %计算文件中图片的数量
% pics=dir('E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0050\*.jpg');
% D=length(pics);
% diffvalue = zeros(1,D);
% tic%开启时间统计
% for t=1:D
%     img=imread([str,num2str(t),'.jpg']); %依次读取每一幅图像
%     value =  EOG(img);
%     diffvalue(1,t) = value;
% end
% toc%关闭时间统计
% figure(1)
% plot(diffvalue);
% hold on


%%
%Tenengrad
%      str='E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0050\'; %待处理数据的保存路径；28里面有好用的，41里有好用的
% %计算文件中图片的数量
% pics=dir('E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0050\*.jpg');
% D=length(pics);
% diffvalue = zeros(1,D);
% tic%开启时间统计
% for t=1:D
%     img=imread([str,num2str(t),'.jpg']); %依次读取每一幅图像
%     value =  Tenengrad_V(img);
%     diffvalue(1,t) = value;
% end
% toc%关闭时间统计
% figure(1)
% plot(diffvalue);
% hold on

%%
%value = Brenner(img)
%      str='E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0050\'; %待处理数据的保存路径；28里面有好用的，41里有好用的
% %计算文件中图片的数量
% pics=dir('E:\aResearch Materials during PhD\2020年\超声图像质量评估\超声数据\已处理图像数据\MI_0050\*.jpg');
% D=length(pics);
% diffvalue = zeros(1,D);
% tic%开启时间统计
% for t=1:D
%     img=imread([str,num2str(t),'.jpg']); %依次读取每一幅图像
%     value =  Brenner(img);
%     diffvalue(1,t) = value;
% end
% toc%关闭时间统计
% figure(1)
% plot(diffvalue);
% hold on



%%
%图像对比度
function value = image_contrast(img)
img=single(img);
img=rgb2gray(img);%变成灰度图
[m,n] = size(img);
img=single(img);
k = 0;
d = mean(mean(img));
for i = 1:m
    for j = 1:n
        k = k+(img(i,j)-d)^2;
    end
end
value = sqrt(k/(m*n));
end


%%
%计算能量模型，使用Ising model和单纯的梯度图
function Isingvalue = Ising_grand(img)
img=single(img);
img=rgb2gray(img);%变成灰度图
[Fx,Fy]= gradient(single(img));
q = 15;
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
Isingvalue = 1.1./(1+exp(-y1+10.2));
end

%%
%能量梯度函数
function value = EOG(img)
img=double(img);
I=rgb2gray(img);%变成灰度图
[M,N] = size(I);
FI = 0;
 for x=2:M-2 
     for y=2:N-2 
          % x方向和y方向的相邻像素灰度值只差的的平方和作为清晰度值
         FI=FI+(I(x+1,y)-I(x,y))*(I(x+1,y)-I(x,y))+(I(x,y+1)-I(x,y))*(I(x,y+1)-I(x,y));
     end 
 end
 value = FI;
end

%%
%Tenengrad 
function value = Tenengrad_V(img)
img=double(img);
I=rgb2gray(img);%变成灰度图
[M,N] = size(I);
GX = 0;   %图像水平方向梯度值
GY = 0;   %图像垂直方向梯度值
FI = 0;   %变量，暂时存储图像清晰度值
T  = 0;   %设置的阈值
 for x=2:M-1 
     for y=2:N-1 
         GX = I(x-1,y+1)+2*I(x,y+1)+I(x+1,y+1)-I(x-1,y-1)-2*I(x,y-1)-I(x+1,y-1); 
         GY = I(x+1,y-1)+2*I(x+1,y)+I(x+1,y+1)-I(x-1,y-1)-2*I(x-1,y)-I(x-1,y+1); 
         SXY= sqrt(GX*GX+GY*GY); %某一点的梯度值
         %某一像素点梯度值大于设定的阈值，将该像素点考虑，消除噪声影响
         if SXY>T 
           FI = FI + SXY*SXY;    %Tenengrad值定义
         end 
     end 
 end 
 value= FI;
end

%%
%Brenner函数
function value = Brenner(img)
img=double(img);
I=rgb2gray(img);%变成灰度图
I=double(I);         %精度存储问题
 [M N]=size(I);     %M等于矩阵行数，N等于矩阵列数；size()获取矩阵行列
 FI=0;        %变量，暂时存储每一幅图像的Brenner值
 for x=1:M-2      %Brenner函数原理，计算相差两个位置的像素点的灰度值
     for y=1:N 
         FI=FI+(I(x+2,y)-I(x,y))*(I(x+2,y)-I(x,y)); 
     end 
 end
 value = FI;
end
