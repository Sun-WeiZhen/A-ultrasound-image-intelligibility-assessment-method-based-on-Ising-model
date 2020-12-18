clc
clear all
close all

%%
%ͼ����----������������
str='E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0041\'; %���������ݵı���·����28�����к��õģ�41���к��õ�
%�����ļ���ͼƬ������
pics=dir('E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0041\*.jpg');
D=length(pics);
diffvalue = zeros(1,D);
tic%����ʱ��ͳ��
for t=1:D
    I=imread([str,num2str(t),'.jpg']); %���ζ�ȡÿһ��ͼ��
    I=single(I);
    I=rgb2gray(I);%��ɻҶ�ͼ
    [C,L]=size(I); %��ͼ��ĳߴ�
    Img_size=C*L; %ͼ�����ص���ܸ���
    G=256; %ͼ��ĻҶȼ�
    H_x=0;
    nk=zeros(G,1);%����һ��G��1�е�ȫ�����
    for i=1:C
        for j=1:L
            Img_level=I(i,j)+1; %��ȡͼ��ĻҶȼ�
            nk(Img_level)=nk(Img_level)+1; %ͳ��ÿ���Ҷȼ����صĵ���
        end
    end
    for k=1:G  %ѭ��
        Ps(k)=nk(k)/Img_size; %����ÿһ�����ص�ĸ���
        if Ps(k)~=0 %������ص�ĸ��ʲ�Ϊ��
            H_x=-Ps(k)*log2(Ps(k))+H_x; %����ֵ�Ĺ�ʽ
        end
    end
    diffvalue(1,t) = H_x;
end
toc%�ر�ʱ��ͳ��
figure(1)
plot(diffvalue);
hold on

%%
%����Isingmodel�ķ���.�������ŵ㣬�㷨����ͼ�������ı仯��Ϊ����
%����ÿ��ͼ���������
% str='E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0050\'; %���������ݵı���·����28�����к��õģ�41���к��õ�
% %�����ļ���ͼƬ������
% pics=dir('E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0050\*.jpg');
% D=length(pics);
% diffvalue = zeros(1,D);
% tic%����ʱ��ͳ��
% for i=1:D
%     img=imread([str,num2str(i),'.jpg']); %���ζ�ȡÿһ��ͼ��
%     Isingvalue = Ising_grand(img);
%     diffvalue(1,i) = Isingvalue;
% end
% toc%�ر�ʱ��ͳ��
% figure(1)
% plot(diffvalue);
% hold on

%%
%����Աȶ�

%      str='E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0050\'; %���������ݵı���·����28�����к��õģ�41���к��õ�
% %�����ļ���ͼƬ������
% pics=dir('E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0050\*.jpg');
% D=length(pics);
% diffvalue = zeros(1,D);
% tic%����ʱ��ͳ��
% for t=1:D
%     img=imread([str,num2str(t),'.jpg']); %���ζ�ȡÿһ��ͼ��
%     value = image_contrast(img);
%     diffvalue(1,t) = value;
% end
% toc%�ر�ʱ��ͳ��
% figure(1)
% plot(diffvalue);
% hold on

%%
%�����ݶȺ���value = EOG(img)
%      str='E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0050\'; %���������ݵı���·����28�����к��õģ�41���к��õ�
% %�����ļ���ͼƬ������
% pics=dir('E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0050\*.jpg');
% D=length(pics);
% diffvalue = zeros(1,D);
% tic%����ʱ��ͳ��
% for t=1:D
%     img=imread([str,num2str(t),'.jpg']); %���ζ�ȡÿһ��ͼ��
%     value =  EOG(img);
%     diffvalue(1,t) = value;
% end
% toc%�ر�ʱ��ͳ��
% figure(1)
% plot(diffvalue);
% hold on


%%
%Tenengrad
%      str='E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0050\'; %���������ݵı���·����28�����к��õģ�41���к��õ�
% %�����ļ���ͼƬ������
% pics=dir('E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0050\*.jpg');
% D=length(pics);
% diffvalue = zeros(1,D);
% tic%����ʱ��ͳ��
% for t=1:D
%     img=imread([str,num2str(t),'.jpg']); %���ζ�ȡÿһ��ͼ��
%     value =  Tenengrad_V(img);
%     diffvalue(1,t) = value;
% end
% toc%�ر�ʱ��ͳ��
% figure(1)
% plot(diffvalue);
% hold on

%%
%value = Brenner(img)
%      str='E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0050\'; %���������ݵı���·����28�����к��õģ�41���к��õ�
% %�����ļ���ͼƬ������
% pics=dir('E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0050\*.jpg');
% D=length(pics);
% diffvalue = zeros(1,D);
% tic%����ʱ��ͳ��
% for t=1:D
%     img=imread([str,num2str(t),'.jpg']); %���ζ�ȡÿһ��ͼ��
%     value =  Brenner(img);
%     diffvalue(1,t) = value;
% end
% toc%�ر�ʱ��ͳ��
% figure(1)
% plot(diffvalue);
% hold on



%%
%ͼ��Աȶ�
function value = image_contrast(img)
img=single(img);
img=rgb2gray(img);%��ɻҶ�ͼ
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
%��������ģ�ͣ�ʹ��Ising model�͵������ݶ�ͼ
function Isingvalue = Ising_grand(img)
img=single(img);
img=rgb2gray(img);%��ɻҶ�ͼ
[Fx,Fy]= gradient(single(img));
q = 15;
Fx(Fx>-q & Fx<0) = 0;
Fx(Fx<q & Fx>0) = 0;
[m,n] = size(Fx);
E_r = [];
for i = 2:m-1
    for j = 2:n-1
        E_r(i,j) = ((Fx(i,j)*Fx(i,j-1)+ Fx(i,j)*Fx(i-1,j) + Fx(i,j)*Fx(i,j+1)...
            + Fx(i,j)*Fx(i+1,j))-Fx(i,j));%��ͼ����ÿ�����ص���������ֵ��
    end
end
Isingvalue = mean(E_r(:));
y1 = 3.4*log(Isingvalue-3.1);
Isingvalue = 1.1./(1+exp(-y1+10.2));
end

%%
%�����ݶȺ���
function value = EOG(img)
img=double(img);
I=rgb2gray(img);%��ɻҶ�ͼ
[M,N] = size(I);
FI = 0;
 for x=2:M-2 
     for y=2:N-2 
          % x�����y������������ػҶ�ֵֻ��ĵ�ƽ������Ϊ������ֵ
         FI=FI+(I(x+1,y)-I(x,y))*(I(x+1,y)-I(x,y))+(I(x,y+1)-I(x,y))*(I(x,y+1)-I(x,y));
     end 
 end
 value = FI;
end

%%
%Tenengrad 
function value = Tenengrad_V(img)
img=double(img);
I=rgb2gray(img);%��ɻҶ�ͼ
[M,N] = size(I);
GX = 0;   %ͼ��ˮƽ�����ݶ�ֵ
GY = 0;   %ͼ��ֱ�����ݶ�ֵ
FI = 0;   %��������ʱ�洢ͼ��������ֵ
T  = 0;   %���õ���ֵ
 for x=2:M-1 
     for y=2:N-1 
         GX = I(x-1,y+1)+2*I(x,y+1)+I(x+1,y+1)-I(x-1,y-1)-2*I(x,y-1)-I(x+1,y-1); 
         GY = I(x+1,y-1)+2*I(x+1,y)+I(x+1,y+1)-I(x-1,y-1)-2*I(x-1,y)-I(x-1,y+1); 
         SXY= sqrt(GX*GX+GY*GY); %ĳһ����ݶ�ֵ
         %ĳһ���ص��ݶ�ֵ�����趨����ֵ���������ص㿼�ǣ���������Ӱ��
         if SXY>T 
           FI = FI + SXY*SXY;    %Tenengradֵ����
         end 
     end 
 end 
 value= FI;
end

%%
%Brenner����
function value = Brenner(img)
img=double(img);
I=rgb2gray(img);%��ɻҶ�ͼ
I=double(I);         %���ȴ洢����
 [M N]=size(I);     %M���ھ���������N���ھ���������size()��ȡ��������
 FI=0;        %��������ʱ�洢ÿһ��ͼ���Brennerֵ
 for x=1:M-2      %Brenner����ԭ�������������λ�õ����ص�ĻҶ�ֵ
     for y=1:N 
         FI=FI+(I(x+2,y)-I(x,y))*(I(x+2,y)-I(x,y)); 
     end 
 end
 value = FI;
end
