clc
clear all
close all
%%
%���ݹ���
%1. ����ͼ������
img = imread('E:\Engineering issues during PhD\Utra2CT_imagedata\Complete ultrasound image data\Ultrasound image data\Clear data\Other organs data\cleanedimg\IM_0053\204.jpg');
%2. ���زο�����
%������ʱ�����������ݵĴ�СΪ0����Ϊ��������״̬
%��ʵ�ϣ����ǲɼ���������֮�󣬵õ��Ľ��������������ʽ
%%
img=rgb2gray(img);%ת���ɻҶ�ͼ
[k,l] = size(img);%����ͼ��ĳߴ磬������Ҫ����һ����������ľ��������ͼ���������жϽ��
[featurecell,featureVector,hogVisualization] = extractCell(double(img),'CellSize',[5 7],'BlockSize',[1 1],'NumBins',6);
figure(1)
imshow(img);
hold on;
% plot(hogVisualization);
% hold on;
%%

%��ô˵�أ��������ǵ���extractcell��������ͼ��ÿ�������cell�����������
%���������������ȸ���������������fraturecell����
%���ȣ���������һ��ʲô˼���أ��������õ�cell��һ��5*7���������ǽ�ͼ����ݶȷ����趨Ϊ6������
%�����������õ����ݶ�ֱ��ͼ��һ��1*6���������������������Ҫ�����ÿ��cell��Ӧ��ֱ��ͼ��
%Ȼ�����Ǹ��ݸ���ÿ�������cellhog�жϸ������ͼ��������ʵ�ֶ�ͼ�����������ĳ�ʼ����
%���ǵĳ�ʼ���жϷ�����ʹ�ø�����������ͱ�׼����ģ������������������׼�Ȼ����ݱ�׼�Χ������������ģ������������
%������Ȼͨ���������ڵ���ʽ�жϡ�
%1. ��Ȼ�����������Ľ���ǣ���ȫ��������ͼ����ݶȸ�������ݶȲ�Ϊ0��ͬʱ�����ǵ�ֱ��ͼ����Ϊ0��������������Ǻ��ٵģ�һ������������������ֱ�Ӷϸ�����Ϊ��������
%��ˣ������������Ͳ���������������Ҫ������ص����ݼ����������ǵ�ƽ��ֵ��
%%
%��ʼ������
%���ȼ���һ���ж��ٸ�����
%����һ���վ������ڴ����Ҫ�����
piex_init = zeros(size(img));
[numh,num_hang,num_lie] = size(featurecell);
%%
%��ʼ��ͼ��
init_image = init_function(img,featurecell,num_hang,num_lie);
%imwrite(init_image,'init_image.jpg');%����ͼ��
figure(2)
imshow(init_image);
hold on;
%%
%������������ģ��
%init_image(init_image>=1) = 0.9999;
%[num_piex1,Ising_qualt] = Isingmodel_mean(init_image);
Ising_qualt = Isingmodel(init_image);%��������������ͼ������ƽ������
%num_piex1 = imrotate(num_piex1,-90);
Ising_qualt = imrotate(Ising_qualt,-90);%��ת
%Ising_qualt(Ising_qualt<0) = Ising_qualt(Ising_qualt<0)*(-1);
%imwrite(Ising_qualt,'Ising_qualt.jpg');%����ͼ��
figure(3)
imshow(Ising_qualt);
hold on

%%
%�㷨���
%������β��ô󴰿ڣ�ÿ��������16*16����ÿ�������ڣ��������������ص�Ising����ģ�͡�

%����ͼ��ߴ�

%%
%�������ʱ���麯��
function [num_piex1,Ising_img] = Isingmodel_mean(img)
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
num_piex = [];
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
            %ISing_avg = mean(win_img(:))+0.05;%ƽ��ֵ���еĻ��͵���
            ISing_avg = 0.65;%ƽ��ֵ���еĻ��͵���
            Num = size(win_img,1)*size(win_img,2);%���㴰����������������
            Numlowavg = sum(sum(win_img<ISing_avg));%��ƽ��ֵС�Ķ��Ǻ�ɫ
            Numhigavg = sum(sum(win_img>ISing_avg));%��ƽ��ֵ��ľ��ǰ�ɫ������Ǽ����ɫ���صĸ�����
            %���numhigavg��ֵ����15*15��һ�룬��ô�����������ж�Ϊ�������򣬵��Ǵ�ʱҪ��С��ƽ��ֵ�ĵط��������ɸ������Ӷ��ı�״̬
            %win_img(win_img<0) = ISing_avg*(-1);%����жϾ������Ƿ��и�������ֵ
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
%�����������Ising model�ĺ�����
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
            ISing_avg = 0.6;%ƽ��ֵ���еĻ��͵���
            Num = size(win_img,1)*size(win_img,2);%���㴰����������������
            %���ǵ������ǣ��ڽ�������Numlowavg��ֵС��NumHalf����Ϊһ���������������������������ġ�
            %��Զ������Numlowavg��ֵ����NumHalf�����������һ����ģ������
            NumHalf = ceil((size(win_img,1)*size(win_img,1))/2);
            Numlowavg = sum(sum(win_img <= ISing_avg));%��ƽ��ֵС���Ǻ�ɫ��
            Numhigavg = sum(sum(win_img > ISing_avg)); %��ƽ��ֵ��ľ��ǰ�ɫ������Ǽ����ɫ���صĸ�����
            win_img(win_img<ISing_avg) = ISing_avg*(-1);%��ͼ��������ֵС��ƽ��ֵʱ�����ж�������״̬Ϊģ��״̬������down�������ȡֵΪ��
            %E = win_img(ceil(size(win_img,1)./2),ceil(size(win_img,2)./2));%��ȡ��ǰ����
            %�������жϴ����������صķ��ţ�����жϵ�ʱ��Ӧ����ƽ��ֵ�жϣ�������0.5�жϣ���Ҫ����һ��
            %�Զ��ж�����������ɫ,�������ؾ��ǵ�ǰ���ء�
            if (Numlowavg <= NumHalf && win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))<0)%����ڰ�ɫ�������������Ǻ�ɫ�ģ���ô�Ͱ���ת������
                img(i,j) = win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))*(-1);
            elseif (Numlowavg > NumHalf && win_img(ceil(size(win_img,1)*0.5),ceil(size(win_img,2)*0.5))>0)%����ں�ɫ�������������ǰ�ɫ�ģ���ôҲҪ����ת������
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
                    if Numlowavg > NumHalf %��ɫ�ٱ�ʾ������������������������ǿ��Ĭ���������ص�����״̬Ϊ��������������ģ����Ǿ�Ҫǿ������Ϊ����
                        if win_img(wi,wj) < 0%������������򣬿�ʼ����ж��������ԣ�����Ǹ���������ת��Ϊƽ��ֵ
                            win_img(wi,wj) = ISing_avg1;
                            E = (-1)*sum(sum(M.*win_img)) - H*(Numhigavg-Num*0.5).*M;%��������ֵ�������ֵ��
                            delta_E = E-E_init;%��������������ı仯�����ǵ������ǣ�������ϵͳ���������𽥱�͵ģ�����delta_E��С����
                            if delta_E < 0
                                win_img(wi,wj) = ISing_avg1 + exp(-i-10) ;%
                            else
                                win_img(wi,wj) = ISing_avg1  ;%.*rand(1).*0.9;
                            end
                        else
                            win_img(wi,wj) = win_img(wi,wj) ;
                        end
                    else     %���������ģ��������ģ������������������ɫ�ĸĳɺ�ɫ�ġ�
                        if win_img(wi,wj) > 0
                            win_img(wi,wj) = (ISing_avg1)*(-1);
                            E = (-1)*sum(sum(M.*win_img)) - H*(Numhigavg-Num*0.5).*M;%��������ֵ�������ֵ
                            delta_E = E-E_init;%��������������ı仯�����ǵ������ǣ�������ϵͳ���������𽥱�͵ģ�����delta_E��С����
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
            %win_img(win_img<0) = ISing_avg*(-1);%����жϾ������Ƿ��и�������ֵ
            %win_img = abs(win_img);
            final_img(i,j) = mean(win_img(:));
        end
    end
end
Ising_img = final_img;
end

%%
%���ȹ���ѭ����ʵ�ֶ�ÿ��cell�ı���
function init_image = init_function(img,featurecell,num_hang,num_lie)
%---------------------------------------------------------------------------
%ģ�����������θ�ֵ.�������㣬���ǿ���ͨ��������������֮�������ֵ��Ϊ���ص�Ȩֵ��
%���ǵ����������������н�����Խ���ʾ���������ļн�ԽС���н�����ԽС��ʾ�������ļн�Խ��
%�����������ķ����غ�ʱ�н�����ȡ���ֵ1�������������ķ�����ȫ�෴�н�����ȡ��Сֵ-1��
%�ڸò��֣�����ȡ�н����ҵľ���ֵ����Ϊ���۱�׼
%���ǵ�Ŀ������Խ������ͼ������Խ�ף��෴��Խģ����ͼ������Խ��
%��������������������������������������������������������������������������������
r = 1;%��1��2��3��4��....16
c = 1;%��1��2��3��4��....16
img_fiff = diff_sun(img);
if img_fiff >1
    qxfeature = [0.353    0.00190    0.00908    0.00550    0.0320    0.527];%���ģ����
else
    qxfeature = [0.0353    0.190    0.00908    0.00550    0.320    0.0527];
end
for i = 1:num_hang%;
    for j = 1:num_lie%
        %���濪ʼ��featurecell����ȡÿ��cell��hog����
        hogi = featurecell(:,i,j);%iÿѭ��һ�Σ��㷨����ȡһ�е�cell����
        hogi = reshape(hogi,1,6);
        %����ʹ��ѵ���õ�SVMģ����Ϊ��ʼ����׼����Ŀǰ�������У�����ʹ�ñ�׼ŷ����þ���������
        D = pdist([hogi;qxfeature],'euclidean');%��׼��ŷ�Ͼ��� (Standardized Euclidean distance)
        C = 1-abs(pdist([qxfeature;hogi], 'cosine'));%Ŀ��֮������Ҽн�
        %ͨ��if�������жϸ�cell���������������������������ģ��
        %��ߵ���Ҫ�̾ỹ�Ǽн�����
        if img_fiff >1
            piex_init(r:r+4,c:c+6) =1*(1./(1./(1+exp(-i*0.1+num_hang./15))+0.8))*((C*1.1+D*0.35)/1.8);%���ģ����
        else
            piex_init(r:r+4,c:c+6) =1*(1./(1./(1+exp(-i*0.1+num_hang./15))+0.8))*((C*1.6+D*0.35)/1.5);%Ŀǰ�Ƚ�����״̬
        end
        %piex_init(r:r+4,c:c+6) =1*(1./(1./(1+exp(-i*0.1+num_hang./15))+0.8))*((C*1.6+D*0.35)/1.5);%Ŀǰ�Ƚ�����״̬
        %piex_init(r:r+4,c:c+6) =1*(1./(1./(1+exp(-i*0.1+num_hang./15))+0.8))*((C*1.1+D*0.35)/1.2);%Ŀǰ�Ƚ�����״̬
        %piex_init(r:r+4,c:c+6) =1*(1./(1./(1+exp(-i*0.1+num_hang./15))+0.8))*((C*1.2+D*0.65)/1.6);%Ŀǰ�Ƚ�����״̬
        %piex_init(r:r+4,c:c+6) =1*(1./(1./(1+exp(-i*0.1+num_hang./15))+0.8))*((1-pdist([qxfeature;hogi], 'cosine')+D*0.75)/1.6);%Ŀǰ�Ƚ�����״̬
        
        c = c + 7;
    end
    r = r+5;
    c = 1;%ÿ������ѭ������֮��Ҫ��������һ����������ӵڶ��п�ʼ������1��ʼ��
end
init_image = piex_init;
end

%%
%�����һ������
function [ Result ] = normalize( Data,lowbound,upbound )
%�����ݾ���Data���й淶��
%�µ��Ͻ���upbound,�µ��½���lowbound
%Ҫ����������ݾ�����ÿһ�б�ʾһ����ά������

msize = size(Data);
Result = zeros(msize(1),msize(2));%�洢���
mins = Data(1,:);%����ÿһά����Сֵ
maxs = Data(1,:);%����ÿһά�����ֵ

%%%%
% ����ÿһά�ȵ���Сֵ�����ֵ
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
%ͼ���뺯��
function img_fiff = diff_sun(img)

[m,n] = size(img);
Y=dct2(img);
I=zeros(m,n);
%��Ƶ����
I(1:m/3,1:n/3)=1;
Ydct=Y.*I;
%��DCT�任
img_filter=uint8(idct2(Ydct));
img_fiff1 = img_filter - img;
img_fiff = mean(img_fiff1(:));
end

%%
%FHOGIM����
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
