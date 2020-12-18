clc
clear all
close all
%%
%����ÿ��ͼ���������
%��������ಿ�ֵģ�Դ������ר���豸
% str='E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0050\'; %���������ݵı���·����28�����к��õģ�41���к��õ�
% %�����ļ���ͼƬ������
% pics=dir('E:\aResearch Materials during PhD\2020��\����ͼ����������\��������\�Ѵ���ͼ������\MI_0050\*.jpg');
%������������ֵģ�Դ����һ������
str='E:\Engineering issues during PhD\Utra2CT_imagedata\Complete ultrasound image data\Ultrasound image data\Clear data\Other organs data\cleanedimg\IM_0009\'; %���������ݵı���·����28�����к��õģ�41���к��õ�
%�����ļ���ͼƬ������
pics=dir('E:\Engineering issues during PhD\Utra2CT_imagedata\Complete ultrasound image data\Ultrasound image data\Clear data\Other organs data\cleanedimg\IM_0009\*.jpg');
D=length(pics);
diffvalue = zeros(1,D);
tic%����ʱ��ͳ��
for i=1:D
    img=imread([str,num2str(i),'.jpg']); %���ζ�ȡÿһ��ͼ��
    Isingvalue = Ising_grand(img);
    diffvalue(1,i) = Isingvalue;
end
toc%�ر�ʱ��ͳ��
diffvalue;
figure(1)
i=1:D;
plot(i,diffvalue,'k');
hold on
%%
%������
%Ising_grand   isingmodel
%imageentropy  ͼ����
%imagecontrast �Աȶ�
%EnergyofGradient �ݶ���������EOG
%Tenengrad1     Tenengrad����
%Brenner1       Brenner����


%%

% final_img = [];
% img = imrotate(single(img1),90);%��ת����
% [m,n] = size(img);
% H =1;%�ⳡ�ų�����ϵ��
%�鿴�Ǹ����㲽�赼��������ȫ��ΪNaN
% for i = 9:m-9
%     for j = 9:n-9
%         if isnan(img(i,j))%�����ǰ����ֵ��NaN����ô�򲻴�������أ�Ĭ��ΪNaN��
%             final_img(i,j) = img(i,j);
%         else
%             win_img = img(i-7:i+7,j-7:j+7);%��ȡ�Ե�ǰ����Ϊ���ĵ�15*15�Ĵ����ڵ����ء�
%             %����win�ڵ�����ֵ��������delta_E�������ء�
% %             for wi = 1:size(win_img,1)
% %                 for wj = 1:size(win_img,2)
% %
% %                 end
% %             end
%             as = mean(win_img(:));
%             %win_img(win_img<0) = ISing_avg*(-1);%����жϾ������Ƿ��и�������ֵ
%             %win_img = abs(win_img);
%             final_img(i,j) = as;
%         end
%     end
% end
% Ising_img = final_img;
% Ising_img = imrotate(Ising_img,-90);%��ת����
% imshow(Ising_img);
% improfile
%%
function Ising_img = Isingmodel_mean(img)
%--------------------------------------------------------------------------
%%����дһ������Ising ����ģ�͵�ͼ��������������
%----------------------------------------
%�ú�������Ҫ˼���ǣ���Σ�������ʵ�����ؼ�������������������Ȼ�Դ��ڵ���ʽ����ͼ��
%���ǵ����ۻ����ǣ�������취�øô����ڵ�ϵͳ�����ĵ���С������ͼ�����������ĵط������ں�ɫ�߿飻��ģ���������ģ���İ߿顣�����ϵͳ�������ﵽ��С����ô��Ȼ����Ҫ��ϵͳ�������ص�״̬ͳһ
%�����������������ĺ�ɫ����ɰ�ɫ���ð�ɫ����ĺ�ɫ����ס�
%��Ҫָ�����ǣ����ǲ���ͼ������Ҽн���Ϊͼ����������ָ�꣬Ҳ����˵��
%��ǰ�����ݶ�ֱ��ͼ���׼��������ֱ��ͼ֮��ļн�ԽС��ʾ���ƶ�Խ�ߣ����ʱ����ֵԽ��
%��ǰ�����ݶ�ֱ��ͼ���׼�����ݶ�ֱ��ͼ֮��н�ԽԽ���ʾ���ض�Խ�ͣ����ʱ����ֵԽС
%����ֵ�ڣ�0��1��֮�䣬ֵԽ��ͼ������ǿ��Խ����
%--------------------------------------------------------------------------
final_img = [];
img = imrotate(img,90);%��ת����
[m,n] = size(img);

H =1;%�ⳡ�ų�����ϵ��
%�鿴�Ǹ����㲽�赼��������ȫ��ΪNaN
for i = 9:m-9
    for j = 9:n-9
        if isnan(img(i,j))%�����ǰ����ֵ��NaN����ô�򲻴�������أ�Ĭ��ΪNaN��
            final_img(i,j) = img(i,j);
        else
            win_img = img(i-7:i+7,j-7:j+7);%��ȡ�Ե�ǰ����Ϊ���ĵ�15*15�Ĵ����ڵ����ء�
            
            %win_img(win_img<0) = ISing_avg*(-1);%����жϾ������Ƿ��и�������ֵ
            %win_img = abs(win_img);
            final_img(i,j) = mean(win_img(:));
        end
    end
end
Ising_img = final_img;
end


%%
%���ݺ���

function Ising_img = Isingmodel(img)
%--------------------------------------------------------------------------
%%����дһ������Ising ����ģ�͵�ͼ��������������
%----------------------------------------
%�ú�������Ҫ˼���ǣ���Σ�������ʵ�����ؼ�������������������Ȼ�Դ��ڵ���ʽ����ͼ��
%���ǵ����ۻ����ǣ�������취�øô����ڵ�ϵͳ�����ĵ���С������ͼ�����������ĵط������ں�ɫ�߿飻��ģ���������ģ���İ߿顣�����ϵͳ�������ﵽ��С����ô��Ȼ����Ҫ��ϵͳ�������ص�״̬ͳһ
%�����������������ĺ�ɫ����ɰ�ɫ���ð�ɫ����ĺ�ɫ����ס�
%��Ҫָ�����ǣ����ǲ���ͼ������Ҽн���Ϊͼ����������ָ�꣬Ҳ����˵��
%��ǰ�����ݶ�ֱ��ͼ���׼��������ֱ��ͼ֮��ļн�ԽС��ʾ���ƶ�Խ�ߣ����ʱ����ֵԽ��
%��ǰ�����ݶ�ֱ��ͼ���׼�����ݶ�ֱ��ͼ֮��н�ԽԽ���ʾ���ض�Խ�ͣ����ʱ����ֵԽС
%����ֵ�ڣ�0��1��֮�䣬ֵԽ��ͼ������ǿ��Խ����
%--------------------------------------------------------------------------
final_img = [];
img = imrotate(img,90);%��ת����
[m,n] = size(img);
H =1;%�ⳡ�ų�����ϵ��
%�鿴�Ǹ����㲽�赼��������ȫ��ΪNaN
for i = 9:m-9
    for j = 9:n-9
        if isnan(img(i,j))%�����ǰ����ֵ��NaN����ô�򲻴�������أ�Ĭ��ΪNaN��
            final_img(i,j) = img(i,j);
        else
            win_img = img(i-7:i+7,j-7:j+7);%��ȡ�Ե�ǰ����Ϊ���ĵ�15*15�Ĵ����ڵ����ء�
            ISing_avg1 = mean(win_img(:));%����ô����ڵ�ƽ��ֵ���������Ϊ
            ISing_avg = mean(win_img(:))+0.25;%ƽ��ֵ���еĻ��͵���
            Num = size(win_img,1)*size(win_img,2);%���㴰����������������
            Numlowavg = sum(sum(win_img>ISing_avg));%��ƽ��ֵ��ľ��ǰ�ɫ������Ǽ����ɫ���صĸ�����
            Numhigavg = sum(sum(win_img<ISing_avg));
            win_img(win_img<ISing_avg) = ISing_avg*(-1);%��ͼ��������ֵС��ƽ��ֵʱ�����ж�������״̬Ϊģ��״̬������down�������ȡֵΪ��
            %E = win_img(ceil(size(win_img,1)./2),ceil(size(win_img,2)./2));%��ȡ��ǰ����
            %�������жϴ����������صķ��ţ�����жϵ�ʱ��Ӧ����ƽ��ֵ�жϣ�������0.5�жϣ���Ҫ����һ��
            %�Զ��ж�����������ɫ,�������ؾ��ǵ�ǰ���ء�
            if (Numlowavg <= Numhigavg && win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))<0)%����ڰ�ɫ�������������Ǻ�ɫ�ģ���ô�Ͱ���ת������
                img(i,j) = win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))*(-1);
            elseif (Numlowavg > Numhigavg && win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))>0)%����ں�ɫ�������������ǰ�ɫ�ģ���ôҲҪ����ת������
                img(i,j) = win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))*(-1);
            else%������������ر�
                img(i,j) = win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5));
            end
            M = img(i,j);%��ȡ��ǰ���ص�����ֵ
            %����win�ڵ�����ֵ��������delta_E�������ء�
            for wi = 1:size(win_img,1)
                for wj = 1:size(win_img,2)
                    E_init = (-1)*sum(sum(M.*win_img)) - H*(Numhigavg-Num*0.5).*M;%û�е�������ֵʱ������
                    %�����ж�����������������ģ������
                    if Numlowavg <= Numhigavg %��ɫ�ٱ�ʾ������������������������ǿ��Ĭ���������ص�����״̬Ϊ��������������ģ����Ǿ�Ҫǿ������Ϊ����
                        if win_img(wi,wj)<0%������������򣬿�ʼ����ж��������ԣ�����Ǹ���������ת��Ϊƽ��ֵ
                            win_img(wi,wj) = ISing_avg;
                            E = (-1)*sum(sum(M.*win_img)) - H*(Numhigavg-Num*0.5).*M;%��������ֵ�������ֵ��
                            delta_E = E-E_init;%��������������ı仯�����ǵ������ǣ�������ϵͳ���������𽥱�͵ģ�����delta_E��С����
                            if delta_E < 0
                                win_img(wi,wj) = ISing_avg - exp(-i-10) ;%
                            else
                                win_img(wi,wj) = ISing_avg - exp(-i-10) ;%.*rand(1).*0.9;
                            end
                        else
                            win_img(wi,wj) = win_img(wi,wj)- exp(-i-10) ;
                        end
                    else     %���������ģ��������ģ������������������ɫ�ĸĳɺ�ɫ�ġ�
                        if win_img(wi,wj) > 0
                            win_img(wi,wj) = (ISing_avg-0.3)*(-1);
                            E = (-1)*sum(sum(M.*win_img)) - H*(Numhigavg-Num*0.5).*M;%��������ֵ�������ֵ
                            delta_E = E-E_init;%��������������ı仯�����ǵ������ǣ�������ϵͳ���������𽥱�͵ģ�����delta_E��С����
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
            %win_img(win_img<0) = ISing_avg*(-1);%����жϾ������Ƿ��и�������ֵ
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
%��Ƶ����
I(1:m/3,1:n/3)=1;
Ydct=Y.*I;
%��DCT�任
img_filter=uint8(idct2(Ydct));
end

function [PSNR] = eval_psnr(img,imgn)
% =================PSNR����
% param ��
%       img:����Ҷ�ͼ��img��imgnͬ�ȴ�С��
%       imgn:����Ҫ���жԱȵĻҶ�ͼ��
%
B = 8;                %����һ�������ö��ٶ�����λ
MAX = 2^B-1;          %ͼ���ж��ٻҶȼ�
[height,width,~] = size(img);
MES = sum(sum((img-imgn).^2))/(height*width);     %������
PSNR = 20*log10(MAX/sqrt(MES));           %��ֵ�����

end

%%
function Isingvalue = Ising_grand(img)
img=rgb2gray(img);%��ɻҶ�ͼ
img = single(img);%�����ȣ��������Ч��
[Fx,Fy]= gradient(img);
q = 13;
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
Isingvalue = (1.1)./(1+exp(-y1+10.2));
end

%%
%ͼ����
function imgentr = imageentropy(img)
img=rgb2gray(img);%��ɻҶ�ͼ
img = single(img);%�����ȣ��������Ч��
[C,L]=size(img); %��ͼ��Ĺ��
Img_size=C*L; %ͼ�����ص���ܸ���
G=256; %ͼ��ĻҶȼ�
H_x=0;
nk=zeros(G,1);%����һ��G��1�е�ȫ�����
for i=1:C
    for j=1:L
        Img_level=img(i,j)+1; %��ȡͼ��ĻҶȼ�
        nk(Img_level)=nk(Img_level)+1; %ͳ��ÿ���Ҷȼ����صĵ���
    end
end
for k=1:G  %ѭ��
    Ps(k)=nk(k)/Img_size; %����ÿһ�����ص�ĸ���
    if Ps(k)~=0; %������ص�ĸ��ʲ�Ϊ��
        H_x=-Ps(k)*log2(Ps(k))+H_x; %����ֵ�Ĺ�ʽ
    end
end
imgentr = H_x;  %��ʾ��ֵ
end

%%
%�Աȶ�
function imgcontrast = imagecontrast(img)
img=rgb2gray(img);%��ɻҶ�ͼ
img = single(img);%�����ȣ��������Ч��
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
img=rgb2gray(img);%��ɻҶ�ͼ
img = single(img);%�����ȣ��������Ч��
[M N]=size(img);
FI=0;
for x=1:M-1
    for y=1:N-1
        % x�����y������������ػҶ�ֵֻ��ĵ�ƽ������Ϊ������ֵ
        FI=FI+(img(x+1,y)-img(x,y))*(img(x+1,y)-img(x,y))+(img(x,y+1)-img(x,y))*(img(x,y+1)-img(x,y));
    end
end
EOG = FI;
end

%%
%Tenengradhanshu
function Tenengrad = Tenengrad1(img)
img=rgb2gray(img);%��ɻҶ�ͼ
img = single(img);%�����ȣ��������Ч��
[M N]=size(img);
%����sobel����gx,gy��ͼ�����������ȡͼ��ˮƽ����ʹ�ֱ������ݶ�ֵ
GX = 0;   %ͼ��ˮƽ�����ݶ�ֵ
GY = 0;   %ͼ��ֱ�����ݶ�ֵ
FI = 0;   %��������ʱ�洢ͼ��������ֵ
T  = 0;   %���õ���ֵ
for x=2:M-1
    for y=2:N-1
        GX = img(x-1,y+1)+2*img(x,y+1)+img(x+1,y+1)-img(x-1,y-1)-2*img(x,y-1)-img(x+1,y-1);
        GY = img(x+1,y-1)+2*img(x+1,y)+img(x+1,y+1)-img(x-1,y-1)-2*img(x-1,y)-img(x-1,y+1);
        SXY= sqrt(GX*GX+GY*GY); %ĳһ����ݶ�ֵ
        %ĳһ���ص��ݶ�ֵ�����趨����ֵ���������ص㿼�ǣ���������Ӱ��
        if SXY>T
            FI = FI + SXY*SXY;    %Tenengradֵ����
        end
    end
end
Tenengrad = FI;
end

%%
%Brenner����
function Brenner = Brenner1(img)
img=rgb2gray(img);%��ɻҶ�ͼ
img = single(img);%�����ȣ��������Ч��
[M N]=size(img);     %M���ھ���������N���ھ���������size()��ȡ��������
FI=0;        %��������ʱ�洢ÿһ��ͼ���Brennerֵ
for x=1:M-2      %Brenner����ԭ�������������λ�õ����ص�ĻҶ�ֵ
    for y=1:N
        FI=FI+(img(x+2,y)-img(x,y))*(img(x+2,y)-img(x,y));
    end
end
Brenner = FI;
end













