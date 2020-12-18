clc
close all
clear all


% img = imread('E:\Engineering issues during PhD\Utra2CT_imagedata\Complete ultrasound image data\Ultrasound image data\Clear data\Other organs data\cleanedimg\IM_0009\1.jpg');
% %img = single(img);%单精度，提高运行效率
% img=rgb2gray(img);%变成灰度图
% img = single(img);%单精度，提高运行效率
% imshow(img)
% x=12:40;
% y1 = 3.4*log(x-3.1);%第一个系数可以有效提高
% y = 1.1./(1+exp(-y1+10.2));
% 
% figure(1)
% plot(y)
% hold on
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%心脏数据集
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load heart.txt;
a = heart;
%MI-0002
is2 = a(1,:);
tropy2 = a(5,:);
tropy2 = mapminmax(tropy2,0,1);
cont2 = a(9,:);
cont2 = mapminmax(cont2,0,1);
eog2 = a(13,:);
eog2 = mapminmax(eog2,0,1);
ten2 = a(17,:);
ten2 = mapminmax(ten2,0,1);
bren2 = a(21,:);
bren2 = mapminmax(bren2,0,1);
%MI-0015
is15 = a(2,:);
tropy15 = a(6,:);
tropy15 = mapminmax(tropy15,0,1);
cont15 = a(10,:);
cont15 = mapminmax(cont15,0,1);
eog15 = a(14,:);
eog15 = mapminmax(eog15,0,1);
ten15 = a(18,:);
ten15 = mapminmax(ten15,0,1);
bren15 = a(22,:);
bren15 = mapminmax(bren15,0,1);
%MI-0028
is28 = a(3,:);
tropy28 = a(7,:);
tropy28 = mapminmax(tropy28,0,1);
cont28 = a(11,:);
cont28 = mapminmax(cont28,0,1);
eog28 = a(15,:);
eog28 = mapminmax(eog28,0,1);
ten28 = a(19,:);
ten28 = mapminmax(ten28,0,1);
bren28 = a(23,:);
bren28 = mapminmax(bren28,0,1);
%MI-0043
is43 = a(4,:);
tropy43 = a(8,:);
tropy43 = mapminmax(tropy43,0,1);
cont43 = a(12,:);
cont43 = mapminmax(cont43,0,1);
eog43 = a(16,:);
eog43 = mapminmax(eog43,0,1);
ten43 = a(20,:);
ten43 = mapminmax(ten43,0,1);
bren43 = a(24,:);
bren43 = mapminmax(bren43,0,1);

%%
%画第一个数据集
figure(1)
plot(is2,'b')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(tropy2,'>-')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(cont2,'*-')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(eog2,'--')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(ten2,'d-')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(bren2,'-.')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
legend('FIAM','Imageentropy','Imagecontrast','EOG','Tenengrad','Brenner') %
title('MI-0002 DataSet');
%%
%画第二个数据集
figure(2)
plot(is15,'b')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(tropy15,'>-')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(cont15,'*-')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(eog15,'--')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(ten15,'d-')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(bren15,'-.')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
legend('FIAM','Imageentropy','Imagecontrast','EOG','Tenengrad','Brenner') %
title('MI-0015 DataSet');
%%
%画第三个数据集
figure(3)
plot(is28,'b')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(tropy28,'>-')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(cont28,'*-')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(eog28,'--')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(ten28,'d-')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(bren28,'-.')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
legend('FIAM','Imageentropy','Imagecontrast','EOG','Tenengrad','Brenner') %
title('MI-0028 DataSet');
%%
%画第四个数据集
figure(4)
plot(is43,'b')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(tropy43,'>-')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(cont43,'*-')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(eog43,'--')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(ten43,'d-')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
plot(bren43,'-.')
xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
ylabel('Quality Index','FontSize',15)
hold on
legend('FIAM','Imageentropy','Imagecontrast','EOG','Tenengrad','Brenner') %
title('MI-0043 DataSet');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%其它器官第一部分
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a = dlmread('organs1.txt');
% %IM-0009
% is2 = a(1,:);
% tropy2 = a(5,:);
% tropy2 = mapminmax(tropy2,0,1);
% cont2 = a(9,:);
% cont2 = mapminmax(cont2,0,1);
% eog2 = a(13,:);
% eog2 = mapminmax(eog2,0,1);
% ten2 = a(17,:);
% ten2 = mapminmax(ten2,0,1);
% bren2 = a(21,:);
% bren2 = mapminmax(bren2,0,1);
% %IM-0011
% is15 = a(2,:);
% tropy15 = a(6,:);
% tropy15 = mapminmax(tropy15,0,1);
% cont15 = a(10,:);
% cont15 = mapminmax(cont15,0,1);
% eog15 = a(14,:);
% eog15 = mapminmax(eog15,0,1);
% ten15 = a(18,:);
% ten15 = mapminmax(ten15,0,1);
% bren15 = a(22,:);
% bren15 = mapminmax(bren15,0,1);
% %IM-0019
% is28 = a(3,:);
% tropy28 = a(7,:);
% tropy28 = mapminmax(tropy28,0,1);
% cont28 = a(11,:);
% cont28 = mapminmax(cont28,0,1);
% eog28 = a(15,:);
% eog28 = mapminmax(eog28,0,1);
% ten28 = a(19,:);
% ten28 = mapminmax(ten28,0,1);
% bren28 = a(23,:);
% bren28 = mapminmax(bren28,0,1);
% %IM-0029
% is43 = a(4,:);
% tropy43 = a(8,:);
% tropy43 = mapminmax(tropy43,0,1);
% cont43 = a(12,:);
% cont43 = mapminmax(cont43,0,1);
% eog43 = a(16,:);
% eog43 = mapminmax(eog43,0,1);
% ten43 = a(20,:);
% ten43 = mapminmax(ten43,0,1);
% bren43 = a(24,:);
% bren43 = mapminmax(bren43,0,1);
% 
% %%
% %画第一个数据集
% figure(1)
% plot(is2,'b')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(tropy2,'>-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(cont2,'*-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(eog2,'--')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(ten2,'d-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(bren2,'-.')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% legend('FIAM','Imageentropy','Imagecontrast','EOG','Tenengrad','Brenner') %
% title('IM-0009 DataSet');
% %%
% %画第二个数据集
% figure(2)
% plot(is15,'b')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(tropy15,'>-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(cont15,'*-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(eog15,'--')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(ten15,'d-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(bren15,'-.')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% legend('FIAM','Imageentropy','Imagecontrast','EOG','Tenengrad','Brenner') %
% title('IM-0011 DataSet');
% %%
% %画第三个数据集
% figure(3)
% plot(is28,'b')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(tropy28,'>-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(cont28,'*-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(eog28,'--')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(ten28,'d-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(bren28,'-.')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% legend('FIAM','Imageentropy','Imagecontrast','EOG','Tenengrad','Brenner') %
% title('IM-0019 DataSet');
% %%
% %画第四个数据集
% figure(4)
% plot(is43,'b')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(tropy43,'>-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(cont43,'*-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(eog43,'--')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(ten43,'d-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(bren43,'-.')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% legend('FIAM','Imageentropy','Imagecontrast','EOG','Tenengrad','Brenner') %
% title('IM-0025 DataSet');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 其它器官第三部分
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 
% a = dlmread('organs2.txt');
% %IM-0048
% is2 = a(1,:);
% tropy2 = a(3,:);
% tropy2 = mapminmax(tropy2,0,1);
% cont2 = a(5,:);
% cont2 = mapminmax(cont2,0,1);
% eog2 = a(7,:);
% eog2 = mapminmax(eog2,0,1);
% ten2 = a(9,:);
% ten2 = mapminmax(ten2,0,1);
% bren2 = a(11,:);
% bren2 = mapminmax(bren2,0,1);
% %IM-0054
% is15 = a(2,1:132);
% tropy15 = a(4,1:132);
% tropy15 = mapminmax(tropy15,0,1);
% cont15 = a(6,1:132);
% cont15 = mapminmax(cont15,0,1);
% eog15 = a(8,1:132);
% eog15 = mapminmax(eog15,0,1);
% ten15 = a(10,1:132);
% ten15 = mapminmax(ten15,0,1);
% bren15 = a(12,1:132);
% bren15 = mapminmax(bren15,0,1);
% 
% 
% %%
% %画第一个数据集
% figure(1)
% plot(is2,'b')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(tropy2,'>-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(cont2,'*-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(eog2,'--')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(ten2,'d-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(bren2,'-.')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% legend('FIAM','Imageentropy','Imagecontrast','EOG','Tenengrad','Brenner') %
% title('IM-0048 DataSet');
% %%
% %画第二个数据集
% figure(2)
% plot(is15,'b')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(tropy15,'>-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(cont15,'*-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(eog15,'--')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(ten15,'d-')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% plot(bren15,'-.')
% xlabel('Number of images','FontSize',15)  %'\bf'是加粗的意思
% ylabel('Quality Index','FontSize',15)
% hold on
% legend('FIAM','Imageentropy','Imagecontrast','EOG','Tenengrad','Brenner') %
% title('IM-0054 DataSet');










