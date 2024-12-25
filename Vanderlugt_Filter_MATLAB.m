clear all;
close all; clc;
warning off;
%%

db = 'olivettifaces.mat';
load(db);
N = size(faces,1);
m=64,n=m,
% Particular_Faces = 2;
n_rows = round(sqrt(400));
n_cols = n_rows;
landa = 633.1e-6 ;
theta = pi/4;
alpha =sin(theta)/landa
%
for i=1:400
    Filename{i}=string("person" + i +".gif");
    img=reshape(faces(:,i),m,n);
    Image{i}=img;
end

for i=1:400
    ii=fix(i/5);
    if rem(ii,2)==0
        iii=2.5*ii+mod(i,5);
        im = Image{i};
        Image1{iii}=im;
    end
end

clear i ii iii img

%%

% figure;clf;set(gcf,'Name','Faces');
%
% for ii=1:400
%     ii;
%     subplot(n_rows,n_cols,ii);
%     %     imagesc(reshape(faces(:,ii),m,n));
%     colormap gray; axis equal tight off;
%     %     title(num2str(ii));
% end

%%
% Vanderlugt filter dataset

x=linspace(-5,5,n);
y=linspace(-5,5,m);
[x,y]=meshgrid(x,y);
U =1*exp(2*pi*j.*alpha.*y);

for i=1:200
   
        im0=Image1{i};
        Fim0=fftshift(fft2(im0));
        FF{i}=Fim0;
        
        A=abs(Fim0);
        Amax=max(max(A));
        A=A./Amax;
        phi=angle(Fim0);
        Fi=A.*exp(j.*phi);
        VF=abs(Fi+U).^2;
        VFa{i}=VF;
        % VF=im2uint8(VF);
        
end
%
%
clear i ii iii

%%
% correlation between random pics and dataset
w =217  %randi([1,26],1,1);

q=Image{w};
%
cor=zeros(1,200);
for k=1:200;
    img=Image1{k}; %VFa{i};
    s = corrcoef(img,q);
    %   s = corrcoef(img(:),q(:));
    cor(k)=abs(s(1,2));
end
xt=find(cor==max(cor));

figure,stem(cor);title("correlation diagram");
xlabel('pic num')
ylabel('correlation coficient')
 
%
Fim0=fftshift(fft2(q));
FF0=Fim0;
figure;imagesc(log(1+abs(FF0)));colormap(gray)
set(gca,'XTick',[],'YTick',[]);
%
A=abs(Fim0);
Amax=max(max(A));
A=A./Amax;
phi=angle(Fim0);
Fi=A.*exp(j.*phi);
VF=abs(Fi+U).^2;
VFa0=VF;
% TL=imadjust(TL,[],[],1);
figure,imagesc(VFa0),colormap(gray);
set(gca,'XTick',[],'YTick',[]);
%
%
H=VFa0.*FF0;
U=real((fft2(H)));


FF1=FF{xt};
H1=VFa0.*FF1;
U1=real((fft2(H1)));

figure;
% clf;set(gcf,'Name','Faces');
% subplot(3,1,1);
imagesc(q),colormap(gray);
axis equal tight off;
set(gca,'XTick',[],'YTick',[]);
% subplot(3,1,2);
figure;imagesc(abs(U));colormap(gray)
set(gca,'XTick',[],'YTick',[]);
axis equal tight off;
%  title(num2str(ii));
% subplot(3,1,3);
figure;imagesc(abs(U1));colormap(gray)
set(gca,'XTick',[],'YTick',[]);
axis equal tight off;

%%
cor0=zeros(1,200);

cor=zeros(1,200);

for i=1:400
    i;
    ii=fix(i/5);
    if rem(ii,2)==1
        iii=2.5*(ii-1)+mod(i,5)+1;
        q=Image{i};
        for k=1:200;
            img=Image1{k}; 
            s = corrcoef(img,q);
            cor0(k)=abs(s(1,2));
        end
%         xt=find(cor==max(cor0));
        cor(iii)=max(cor0);
    end
end

figure,stem(cor);title("correlation");
xlabel('pic num')
ylabel('correlation coficient')

erorcor=(1-cor)*100;
figure,stem(erorcor);title("error");
xlabel('pic num')
ylabel('error percent')

M = mean(erorcor)