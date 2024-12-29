% Vanderlugt Filter Implementation for Face Recognition
% Authors: Nahal Akbari, Hassan Kaatuzian
% Description: MATLAB implementation of Vanderlugt filter for facial recognition
% using optical correlation techniques.

clear all;
close all; clc;
warning off;

%% Initialize Parameters and Load Data
% Load the ORL face database
db = 'olivettifaces.mat';
load(db);
N = size(faces,1);
m=64; n=m;  % Image dimensions (64x64 pixels)
n_rows = round(sqrt(400));  % For display purposes
n_cols = n_rows;

% Set optical system parameters
lambda = 633.1e-6;  % HeNe laser wavelength (mm)
theta = pi/4;       % Reference beam angle
alpha = sin(theta)/lambda;  % Spatial frequency

%% Prepare Image Arrays
% Initialize cell arrays for storing images
Image = cell(1,400);    % All images
Image1 = cell(1,200);   % Training set images
    
% Reshape all images to 64x64
for i=1:400
    img = reshape(faces(:,i),m,n);
    Image{i} = img;
end

% Create training set (every other set of 5 images)
for i=1:400
    ii = fix(i/5);
    if rem(ii,2)==0  % Select even sets
        iii = 2.5*ii + mod(i,5);
        im = Image{i};
        Image1{iii} = im;
    end
end
clear i ii iii img

%% Generate Vanderlugt Filter
% Create coordinate grid for reference wave
x = linspace(-5,5,n);
y = linspace(-5,5,m);
[x,y] = meshgrid(x,y);
U = 1*exp(2*pi*j.*alpha.*y);  % Reference wave

% Generate filters for each training image
for i=1:200
    if ~isempty(Image1{i})
        im0 = Image1{i};
        Fim0 = fftshift(fft2(im0));  % Fourier transform
        FF{i} = Fim0;  % Store Fourier transform
        
        % Normalize amplitude
        A = abs(Fim0);
        Amax = max(max(A));
        A = A./Amax;
        
        % Calculate phase
        phi = angle(Fim0);
        Fi = A.*exp(j.*phi);
        
        % Create Vanderlugt filter
        VF = abs(Fi+U).^2;
        VFa{i} = VF;
    end
end
clear i ii iii

%% Perform Correlation Analysis
% Select test image
w = 217;  % Test image index
q = Image{w};

% Calculate correlation with all training images
cor = zeros(1,200);
for k=1:200
    img = Image1{k};
    s = corrcoef(img,q);
    cor(k) = abs(s(1,2));
end
xt = find(cor==max(cor));  % Find best match

% Plot correlation results
figure,stem(cor);title("correlation diagram");
xlabel('pic num')
ylabel('correlation coefficient')

%% Process Test Image
% Compute Fourier transform of test image
Fim0 = fftshift(fft2(q));
FF0 = Fim0;
figure;imagesc(log(1+abs(FF0)));colormap(gray)  % Display Fourier spectrum
set(gca,'XTick',[],'YTick',[]);

% Generate Vanderlugt filter for test image
A = abs(Fim0);
Amax = max(max(A));
A = A./Amax;
phi = angle(Fim0);
Fi = A.*exp(j.*phi);
VF = abs(Fi+U).^2;
VFa0 = VF;

% Display filter
figure,imagesc(VFa0),colormap(gray);
set(gca,'XTick',[],'YTick',[]);

%% Compute Final Correlation
% Perform correlation operations
H = VFa0.*FF0;
U = real((fft2(H)));
FF1 = FF{xt};
H1 = VFa0.*FF1;
U1 = real((fft2(H1)));

% Display results
figure;
imagesc(q),colormap(gray);  % Original image
axis equal tight off;
set(gca,'XTick',[],'YTick',[]);

figure;imagesc(abs(U));colormap(gray)  % Correlation output 1
set(gca,'XTick',[],'YTick',[]);
axis equal tight off;

figure;imagesc(abs(U1));colormap(gray)  % Correlation output 2
set(gca,'XTick',[],'YTick',[]);
axis equal tight off;

%% Calculate Error Rates
% Compute correlations for all test images
cor0 = zeros(1,200);
cor = zeros(1,200);
for i=1:400
    ii = fix(i/5);
    if rem(ii,2)==1  % Select odd sets for testing
        iii = 2.5*(ii-1)+mod(i,5)+1;
        q = Image{i};
        for k=1:200
            img = Image1{k}; 
            s = corrcoef(img,q);
            cor0(k) = abs(s(1,2));
        end
        cor(iii) = max(cor0);
    end
end

% Plot final results
figure,stem(cor);title("correlation");
xlabel('pic num')
ylabel('correlation coefficient')
erorcor = (1-cor)*100;  % Calculate error percentage
figure,stem(erorcor);title("error");
xlabel('pic num')
ylabel('error percent')
M = mean(erorcor)  % Calculate mean error
