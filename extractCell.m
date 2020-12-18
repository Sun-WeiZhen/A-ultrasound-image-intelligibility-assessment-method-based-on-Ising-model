function [hogfeateur,features, varargout] = extractCell(I,varargin)
%extractHOGFeatures提取HOG特征。
%features = extractHOGFeatures（I）从真彩色或灰度图像I中提取HOG特征，并以1×N向量返回特征。 
%这些特征对图像中区域的局部形状信息进行编码，可用于许多任务，包括分类，检测和跟踪，图像质量评估。
%
%HOG特征长度N基于图像大小和下面列出的参数值。
%参考 <a href="matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'extractHOGFeatures')" >documentation</a> for more information. 
%
%[features, validPoints] = extractHOGFeatures(I, points) 返回在图像I中的点位置周围提取的HOG特征。
%该函数还返回validPoints，其中包含输入点位置，其周围的[CellSize.* BlockSize]区域完全包含在图像I中。 
%可以将输入点指定为[xy]坐标，SURFPoints，cornerPoints，MSERRegions或BRISKPoints的M×2矩阵。 
%与这些点关联的任何比例尺信息都将被忽略。 validPoints的类别与输入点相同。.
%
%  [..., visualization] = extractHOGFeatures(I, ...) 返回可以使用plot（visualization）显示的HOG特征，用于HOG特征的可视化。
%
%  [...] = extractHOGFeatures(..., Name, Value) 指定其他名称/值对，如下所述：
%
%  'CellSize'     是2元素向量，用于指定HOG单元的大小（以像素为单位）。 
%                 选择较大的cell尺寸以捕获大规模的空间信息，但要以牺牲小型细节为代价。
%                 
%                 Default: [5 8]
%
%  'BlockSize'    是2元素向量，用于指定块中的cell的数量。 
%                 大的块大小值会降低使局部照明变化最小化的能力。
%
%                 Default: [2 2]
%
%  'BlockOverlap' 是2元素向量，用于指定相邻块之间的重叠cell数量。 
%                 选择至少为块大小一半的重叠部分，以确保足够的对比度归一化。 
%                 较大的重叠值可以以增加的特征向量大小为代价捕获更多信息。 
%                 提取点位置周围的HOG要素时，此属性无效。
% 
%                 Default: ceil(BlockSize/2)
%                  
%  'NumBins'      一个正的标量，指定方向直方图中的bin数。 增加此值可编码更精细的方向细节。
%                 
%                 Default: 9
%
%'UseSignedOrientation'是一个逻辑标量。 如果为true，则将方向值合并到-180至180度之间的均匀间隔的合并bins中。 
%                      否则，将定向值合并到0到180之间，其中将小于0的theta值放入theta + 180 bin中。 
%                      使用带符号的方向可以帮助区分图像区域中的明暗过渡与暗明过渡。
%
%
%     Default: false
%
% Class Support
% -------------
%输入的图像我可以是uint8，int16，double，single或逻辑，它必须是实数和非稀疏的。
%POINTS可以是SURFPoints，cornerPoints，MSERRegions，BRISKPoints，int16，uint16，int32，uint32，单或双。
%
%
% 调用函数的示例1-从图像中提取HOG特征。
% -----------------------------------------------
%
%    I1 = imread('gantrycrane.png');
%    [hog1, visualization] = extractHOGFeatures(I1,'CellSize',[32 32]);
%    subplot(1,2,1);
%    imshow(I1);
%    subplot(1,2,2);
%    plot(visualization);
%
% 调用函数的示例2-提取拐角点附近的HOG特征。
% ------------------------------------------------------
%
%    I2 = imread('gantrycrane.png');
%    corners   = detectFASTFeatures(rgb2gray(I2));
%    strongest = selectStrongest(corners, 3);
%    [hog2, validPoints, ptVis] = extractHOGFeatures(I2, strongest);
%    figure;
%    imshow(I2); hold on;
%    plot(ptVis, 'Color','green');
% 
% 其他的部分，请参见extractFeatures，extractLBPFeatures，detectHarrisFeatures，
%detectFASTFeature，detectMinEigenFeatures，detectSURFFeature，
%detectMSERFeatures，detectBRISKFeatures


%#codegen
%#ok<*EMCA>

notCodegen = isempty(coder.target);%判断代码生成目标是否为指定目标

[points, isPoints, params, maxargs] = parseInputs(I,varargin{:});

% 检查输出数量（outputs numbers）
if notCodegen
    nargoutchk(0,maxargs);%检查输出
else    
    checkNumOutputsForCodegen(nargout, maxargs);%如果生成代码，则不允许HOG可视化
end

if isPoints
    %检查调用哪种类型的特征提取器，一共两个，从点提取和从整张图像提取
    [features, validPoints] = extractHOGFromPoints(I, points, params);
    
    if nargout >= 2%计算该函数输出参数的个数
        varargout{1} = validPoints;
    end
    
    %nargout的作用是在matlab中定义一个函数时， 
    %在函数体内部， nargout指出了输出参数的个数（nargin指出了输入参数的个数）。 特别是在利用了可变参数列表的函数中， 用nargout获取输出参数个数很方便。
    
    if notCodegen
        if nargout == 3
            params.Points = validPoints;
            varargout{2}  = vision.internal.hog.Visualization(features, params);
        end
    end
else   
   
    [hogfeateur,features] = extractHOGFromImage(I, params);  
   
    if notCodegen
        if nargout == 3%原本是2
            
            varargout{1} = vision.internal.hog.Visualization(features, params);
            
        end
    end
end
 
% -------------------------------------------------------------------------
% 从整个图像中提取HOG特征 
% -------------------------------------------------------------------------
function [hogfeateur,features] = extractHOGFromImage(I, params)
[gMag, gDir] = hogGradient(I);%计算图像梯度幅值和梯度角度，函数在下面386行给出

[gaussian, spatial] = computeWeights(params);%计算高斯和空间权重，该函数在375行给出，

[hogfeateur, features] = extractHOG(gMag, gDir, gaussian, spatial, params);

% -------------------------------------------------------------------------
% 根据点位置提取HOG特征
% -------------------------------------------------------------------------
function [features, validPoints] = extractHOGFromPoints(I, points, params)

featureClass = coder.internal.const('single');
uintClass    = coder.internal.const('uint32');

blockSizeInPixels = params.CellSize.*params.BlockSize;

% compute weights
[gaussian, spatial] = computeWeights(params);

if ~isnumeric(points)
    xy = points.Location;
else
    xy = points;
end

featureSize = vision.internal.hog.getFeatureSize(params);

halfSize = (single(blockSizeInPixels) - mod(single(blockSizeInPixels),2))./2;

roi = [1 1 blockSizeInPixels]; % [r c height width]

numPoints       = cast(size(xy,1), uintClass);
validPointIdx   = zeros(1, numPoints , uintClass);
validPointCount = zeros(1, uintClass);

features = zeros(numPoints, featureSize, featureClass);
for i = 1:numPoints
    
    % ROI 以点位置为中心
    roi(1:2) = cast(round(xy(i,[2 1])), featureClass) - halfSize;
    
    % 仅在图像中完全包含ROI时处理
    if all(roi(1:2) >= 1) && ...
            roi(1)+roi(3)-1 <= params.ImageSize(1) && ...
            roi(2)+roi(4)-1 <= params.ImageSize(2)
        
        validPointCount = validPointCount + 1;
        
        [gMag, gDir] = hogGradient(I, roi);
               
        hog = extractHOG(gMag, gDir, gaussian, spatial, params);
        
        features(validPointCount,:) = hog(:);
        validPointIdx(validPointCount) = i; % 存储有效索引
    end
    
end

features = features(1:validPointCount,:);

validPoints = extractValidPoints(points, validPointIdx(1:validPointCount));

% -------------------------------------------------------------------------
% 在给定梯度幅值和方向的情况下提取HOG特征
% -------------------------------------------------------------------------
function [cellhog,hog] = extractHOG(gMag, gDir, gaussianWeights, weights, params)
    %----------------------------------------------------------------------
    %
    %这里输入的gMag和gDir都是与图像img相同尺寸的大小，里面每个数据分别是该图像每个像素的梯度幅值和梯度方向。因此是个矩阵
    %
    %----------------------------------------------------------------------
% if isempty(coder.target)         
%     hog = visionExtractHOGFeatures(gMag, gDir, gaussianWeights, params, weights); 
%     cellhog = rand(2,3);
% else
    featureClass = 'single';%定义特征类型为单精度
    
    if params.UseSignedOrientation%默认为0，因为我们习惯了使用180度
        % 定义gDir的范围在 [0 360]之间
        histRange = single(360);%单精度型变量
    else
        % 把方向转换成无符号型，范围在 [0 180]之间
        histRange = single(180);%单精度型变量
    end
    
    %如果 gDir 范围是 [-180 180], 转换到  [0 180] 或者 [0 360]之间
    negDir = gDir < 0;%真为1，假为0
    gDir(negDir) = histRange + gDir(negDir);
    
    % 为所有的cells定位方向bin
    binWidth = histRange/cast(params.NumBins, featureClass);%例如，计算在180度的情况下，有9个bin，则每个bin的宽度是20度。
    [x1, b1] = computeLowerHistBin(gDir, binWidth);
    %利用该公式计算出当前像素的梯度方向在bin中哪个位置。x1表示当前梯度方向角度整数角度中间值，
    %比如第一个bin是0-20度，那么这个值就是10，b1表示该角度在bin中的第几格。
    wDir = 1 - (gDir - x1)./binWidth;%方向权重，与中间值偏离越大，权重越小。这是很自然的。不理解的话可以自己带入参数试试。这是一个与图像大小相同的矩阵
    
    blockSizeInPixels = params.CellSize.*params.BlockSize;%（8，8）*（2，2），表示block包含像素的数量（16，16），像素量64
    blockStepInPixels = params.CellSize.*(params.BlockSize - params.BlockOverlap);%表示block每次移动几个像素。
    
    r = 1:blockSizeInPixels(1);%行1，2，3，4，....16
    c = 1:blockSizeInPixels(2);%列1，2，3，4，....16
    
    nCells  = params.BlockSize;%每个block的尺寸，用于计算每个block中cell的个数 2*2
    nBlocks = vision.internal.hog.getNumBlocksPerWindow(params);%每个窗口中block的数量
    size(nBlocks)
    numCellsPerBlock = nCells(1)*nCells(2);%每个block中cell的个数,4
    
    %首先计算每个block中有都少个bin
    hog = coder.nullcopy(zeros([params.NumBins*numCellsPerBlock, nBlocks],featureClass)); %（bin的数量乘以每个block中cell的数量，窗口中block的数量）。
    %hog的存储思想是：每个bin占九个单位，每个cell一个bin，每个block有4个cell，。每列一个block的bin
    %（9*4,n），n是每个窗口中block的个数。每个cell中有9个bin，每个block中有4个cell，所有每个block中有36个b空间，分别是4个cell的bin
    %nBlocks = 15*7
    %定义的这个hog，其思想就是，每一列就是一个block所包含的bin的数量，1到9是cell（1，1）的bin，10到18是cell(1,2)的bin，19到27是cell(2,1)的bin，28到36是cell(2,2)的bin
    % 扫描窗口内的所有block，起到遍历图像的目的
    for j = 1:nBlocks(2)%;列       
        for i = 1:nBlocks(1)%行            
            %对方向权重进行线性插值加权
            wz1 = wDir(r,c); %其实就是  wz1 =  wDir.表示当前梯度方向属于该bin的权重。这是一个与block大小相同的矩阵
            w = trilinearWeights(wz1, weights); %                       
            % 对幅值进行高斯滤波加权
            m = gMag(r,c) .* gaussianWeights; %对像素幅值进行高斯平滑处理。这是一个与block大小相同的矩阵           
            % interpolate magnitudes for binning 对梯度幅值进行插值并存到bin中
            
            %下面的步骤完成了将梯度幅值和梯度方向进行加权融合的任务，从而将会两个数据整合成一个数据，
            mx1y1z1 = m .* w.x1_y1_z1;
            mx1y1z2 = m .* w.x1_y1_z2;
            mx1y2z1 = m .* w.x1_y2_z1;
            mx1y2z2 = m .* w.x1_y2_z2;
            mx2y1z1 = m .* w.x2_y1_z1;
            mx2y1z2 = m .* w.x2_y1_z2;
            mx2y2z1 = m .* w.x2_y2_z1;
            mx2y2z2 = m .* w.x2_y2_z2;            
            %--------------------------------------------------------------
            %下面开始处理该存储在哪个bin里面
            %--------------------------------------------------------------
            orientationBins = b1(r,c);%在像素的角度列表中提取某个像素的角度对应的区间号            
            % 将块直方图初始化为零
            h = zeros(params.NumBins+2, nCells(1)+2, nCells(2)+2, featureClass);%单精度，h用于存储每个block中的bin。
            %--------------------------------------------------------------
            %这边需要着重搞懂直方图的存储策略
            %h被定义成一个由10个11*10的三维矩阵，数据类型为单精度类型，其中第三个10表示角度区间位置：1，2，3，4，5，6，7，8，9，10
            %1表示：0-20度，2表示：20-40度，以此类推.........
            %其中，每个11*10的矩阵用于存储
            %--------------------------------------------------------------            
            % 这里也是逐行读取block中的每个像素。
            for x = 1:blockSizeInPixels(2)%1：16,用列数，表示横着走，n
                cx = weights.cellX(x);%判断这个像素属于哪个cell
                for y = 1:blockSizeInPixels(1)%1：16
                    z  = orientationBins(y,x);%这个像素的梯度方向属于哪个bin，这里判断出来的角度事实际属于的带宽内，而z+1是比这个z大的另一个带宽。
                    %因为算法需要在两个相邻bin内进行加权，根据距离进行加权。
                    cy = weights.cellY(y);%1，2，3
                    
                    h(z,   cy,   cx  ) = h(z,   cy,   cx  ) + mx1y1z1(y,x);
                    h(z+1, cy,   cx  ) = h(z+1, cy,   cx  ) + mx1y1z2(y,x);
                    h(z,   cy+1, cx  ) = h(z,   cy+1, cx  ) + mx1y2z1(y,x);
                    h(z+1, cy+1, cx  ) = h(z+1, cy+1, cx  ) + mx1y2z2(y,x);
                    h(z,   cy,   cx+1) = h(z,   cy,   cx+1) + mx2y1z1(y,x);
                    h(z+1, cy,   cx+1) = h(z+1, cy,   cx+1) + mx2y1z2(y,x);
                    h(z,   cy+1, cx+1) = h(z,   cy+1, cx+1) + mx2y2z1(y,x);
                    h(z+1, cy+1, cx+1) = h(z+1, cy+1, cx+1) + mx2y2z2(y,x);
                end
            end
            
            % h装填，这里这样操作是因为上面创建h时在x和y上+2，之所以加2，是因为我们在计算像素属于哪个cell时，出现1，2，3，然而3不是我们想要的。
            h(2,:,:)     = h(2,:,:)     + h(end,:,:);
            h(end-1,:,:) = h(end-1,:,:) + h(1,:,:);
            
            % 仅保留块直方图的有效部分
            h = h(2:end-1,2:end-1,2:end-1);
            
            %正则化并将块添加到特征向量         
            hog(:,i,j) = normalizeL2Hys(h(:));%从代码上观察，每个h表示一个block中的所有bin，在hog中，每个j表示一页，每一页的矩阵便是一个block的所有bin值，       
            r = r + blockStepInPixels(1);
        end
        r = 1:blockSizeInPixels(1);
        c = c + blockStepInPixels(2);
    end
    cellhog = hog;
    hog = reshape(hog, 1, []);%把hog的特征reshape一个一行N列的向量，用于到SVM等算法中训练
%end

% -------------------------------------------------------------------------
% 使用L2-Hys归一化向量
% -------------------------------------------------------------------------
function x = normalizeL2Hys(x)
classToUse = class(x);
x = x./(norm(x,2) + eps(classToUse)); % L2 norm
x(x > 0.2) = 0.2;                     % 大于0.2的都修剪成 0.2
x = x./(norm(x,2) + eps(classToUse)); % 重复L2正则化

% -------------------------------------------------------------------------
% 计算cell上空间直方图的插值权重，其实就是双线性插值
%双线性插值的思想是：分别求出来点（x，y）周围四个点的权重，就可以根据这些权重计算这个点的坐标
% -------------------------------------------------------------------------
function weights = spatialHistWeights(params)
%--------------------------------------------------------------------------
%这个函数的主要作用是对图像上的点进行双线性空间插值，求得每个像素之间的差值权重，
%主要是多四个点中间点坐标进行估计，
%首先进行x轴方向的插值，然后进行Y轴方向的插值。但是i线性插值的结果与差值顺序无关，先计算那个方向是无所谓的
%
%需要指出的是，为了减少混叠，选票在方向和位置上都是在相邻的bins中心之间进行双线性内插。
%因此针对该函数的使用，分别在计算像素梯度方向和像素位置的时候进行插值
%--------------------------------------------------------------------------

%空间直方图权重
% 针对（x，y）周围的4个点计算2D插值权重
%
% (x1,y1) o---------o (x2,y1)
%         |         |
%         |  (x,y)  |
%         |         |
% (x1,y2) o---------o (x2,y2)
%
% （x，y）是HOG块block内的像素中心
%
% (x1,y1); (x2,y1); (x1,y2); (x2,y2) 是一个块内的cell中心

%计算当前block所在的位置覆盖的图像区域的范围
width  = single(params.BlockSize(2)*params.CellSize(2));%块的宽乘以cell的宽，
height = single(params.BlockSize(1)*params.CellSize(1));%块的高乘以cell的高，

x = 0.5:1:width;
y = 0.5:1:height;

[x1, cellX1] = computeLowerHistBin(x, params.CellSize(2));%function [x1, b1] = computeLowerHistBin(x, binWidth)
[y1, cellY1] = computeLowerHistBin(y, params.CellSize(1));

wx1 = 1 - (x - x1)./single(params.CellSize(2));%8
wy1 = 1 - (y - y1)./single(params.CellSize(1));

%使用结构体，这是双线性插值的简化形式，这种形式是默认在四个已知点坐标分别为 (0, 0)、(0, 1)、(1, 0) 和 (1, 1)的情况下的特殊情况，那么插值公式就可以化简为
%参考网页https://blog.csdn.net/xjz18298268521/article/details/51220576
weights.x1y1 = wy1' * wx1;
weights.x2y1 = wy1' * (1-wx1);
weights.x1y2 = (1-wy1)' * wx1;
weights.x2y2 = (1-wy1)' * (1-wx1);

% also store the cell indices  也是存储cell索引
weights.cellX = cellX1;
weights.cellY = cellY1;

% -------------------------------------------------------------------------
% 计算三重线性权重，取得是空间中的八个点，计算这个八个点之间任意一个点的位置
% -------------------------------------------------------------------------
function weights = trilinearWeights(wz1, spatialWeights)

% 在使用前定义结构字段
weights.x1_y1_z1 = coder.nullcopy(wz1);
weights.x1_y1_z2 = coder.nullcopy(wz1);
weights.x2_y1_z1 = coder.nullcopy(wz1);
weights.x2_y1_z2 = coder.nullcopy(wz1);
weights.x1_y2_z1 = coder.nullcopy(wz1);
weights.x1_y2_z2 = coder.nullcopy(wz1);
weights.x2_y2_z1 = coder.nullcopy(wz1);
weights.x2_y2_z2 = coder.nullcopy(wz1);

weights.x1_y1_z1 = wz1 .* spatialWeights.x1y1;
weights.x1_y1_z2 = spatialWeights.x1y1 - weights.x1_y1_z1;
weights.x2_y1_z1 = wz1 .* spatialWeights.x2y1;
weights.x2_y1_z2 = spatialWeights.x2y1 - weights.x2_y1_z1;
weights.x1_y2_z1 = wz1 .* spatialWeights.x1y2;
weights.x1_y2_z2 = spatialWeights.x1y2 - weights.x1_y2_z1;
weights.x2_y2_z1 = wz1 .* spatialWeights.x2y2;
weights.x2_y2_z2 = spatialWeights.x2y2 - weights.x2_y2_z1;

% -------------------------------------------------------------------------
% 计算小于或等于x的,最接近bin中心的x1
% -------------------------------------------------------------------------
function [x1, b1] = computeLowerHistBin(x, binWidth)
% Bin 的索引
width    = single(binWidth);%转换成单精度
invWidth = 1./width;%求倒数
bin      = floor(x.*invWidth - 0.5);%floor函数朝负无穷大方向取整，例如-1.2，取值-2。
% Bin 的中心 x1
x1 = width * (bin + 0.5);
% 加2以获得基于1的索引
b1 = int32(bin + 2);

% -------------------------------------------------------------------------
%计算高斯和空间权重
% -------------------------------------------------------------------------
function [gaussian, spatial] = computeWeights(params)
blockSizeInPixels = params.CellSize.*params.BlockSize;%
gaussian = gaussianWeights(blockSizeInPixels);%高斯滤波，平滑图像，凸显边缘
spatial  = spatialHistWeights(params);

% -------------------------------------------------------------------------
% 使用差分滤波器[-1 0 1]进行梯度计算。 使用前向差计算图像边界处的渐变。 
% 从正X轴逆时针测量，梯度方向在-180至180度之间。
% -------------------------------------------------------------------------
function [gMag, gDir] = hogGradient(img,roi)

if nargin == 1  %nargin 是用来判断输入变量个数的函数，这里的作用是判断是否只对图像的ROI区域进行求HOG特征。
    roi = [];    
    imsize = size(img);
else
    imsize = roi(3:4);
end

img = single(img);%图像类型转换成单精度

if ndims(img)==3%如果图像时三维彩色图像
    rgbMag = zeros([imsize(1:2) 3], 'like', img);
    rgbDir = zeros([imsize(1:2) 3], 'like', img);
    
    for i = 1:3
        %
        [rgbMag(:,:,i), rgbDir(:,:,i)] = computeGradient(img(:,:,i),roi);
    end
    
    % 找到每个像素的最大颜色渐变
    [gMag, maxChannelIdx] = max(rgbMag,[],3);
    
    % 从幅度最大的位置提取梯度方向
    sz = size(rgbMag);
    [rIdx, cIdx] = ndgrid(1:sz(1), 1:sz(2));
    ind  = sub2ind(sz, rIdx(:), cIdx(:), maxChannelIdx(:));
    gDir = reshape(rgbDir(ind), sz(1:2));
else
    [gMag,gDir] = computeGradient(img,roi);
end

% -------------------------------------------------------------------------
% 该函数仅仅给出了TOI区域内图像梯度Ix和Iy的计算方法，在下面的函数中会直接调用。
% -------------------------------------------------------------------------
function [gx, gy] = computeGradientROI(img, roi)
operator_gx = [-1,0,1] ;
operator_gy = operator_gx' ;
img    = single(img);
imsize = size(img);

% roi is [r c height width]
rIdx = roi(1):roi(1)+roi(3)-1;
cIdx = roi(2):roi(2)+roi(4)-1;

imgX = coder.nullcopy(zeros([roi(3)   roi(4)+2], 'like', img)); %#ok<NASGU>
imgY = coder.nullcopy(zeros([roi(3)+2 roi(4)  ], 'like', img)); %#ok<NASGU>

% 如果ROI在图像边框上，则复制边框像素. 
if rIdx(1) == 1 || cIdx(1)==1  || rIdx(end) == imsize(1) ...
        || cIdx(end) == imsize(2)
    
    if rIdx(1) == 1
        padTop = img(rIdx(1), cIdx);
    else
        padTop = img(rIdx(1)-1, cIdx);
    end
    
    if rIdx(end) == imsize(1)
        padBottom = img(rIdx(end), cIdx);
    else
        padBottom = img(rIdx(end)+1, cIdx);
    end
    
    if cIdx(1) == 1
        padLeft = img(rIdx, cIdx(1));
    else
        padLeft = img(rIdx, cIdx(1)-1);
    end
    
    if cIdx(end) == imsize(2)
        padRight = img(rIdx, cIdx(end));
    else
        padRight = img(rIdx, cIdx(end)+1);
    end
    
    imgX = [padLeft img(rIdx,cIdx) padRight];
    imgY = [padTop; img(rIdx,cIdx);padBottom];
else  
    imgX = img(rIdx,[cIdx(1)-1 cIdx cIdx(end)+1]);
    imgY = img([rIdx(1)-1 rIdx rIdx(end)+1],cIdx);
end

gx = conv2(imgX, operator_gx, 'valid');
gy = conv2(imgY, operator_gy, 'valid');

% -------------------------------------------------------------------------
%该函数，首先给出了整张图像的梯度计算方法，然后使用这个函数computeGradientROI（）计算需要计算ROI区域时的梯度。最后给出了梯度幅值和梯度方向的计算方法
%
% -------------------------------------------------------------------------
function [gMag,gDir] = computeGradient(img,roi)
operator_gx = [-1,0,1] ;
operator_gy = operator_gx' ;
if isempty(roi)
    %如果不用计算ROI
    %gx和gy是用来存放图像在x轴方向和y轴方向上的梯度的
    gx = zeros(size(img), 'like', img);%返回像图像img这样的矩阵
    gy = zeros(size(img), 'like', img);%返回像图像img这样的矩阵
    
    gx(:,2:end-1) = conv2(img, operator_gx, 'valid');%卷积滤波，img是被卷积的图像，[-1,0,1]是卷积核，其实就是求梯度
    gy(2:end-1,:) = conv2(img, operator_gy, 'valid');
    
    % 边界上的梯度
    gx(:,1)   = img(:,2)   - img(:,1);
    gx(:,end) = img(:,end) - img(:,end-1);
    gy(1,:)   = img(2,:)   - img(1,:);
    gy(end,:) = img(end,:) - img(end-1,:);
else
    %计算ROI则需要使用专门的梯度计算函数
    [gx, gy] = computeGradientROI(img, roi);
end

% 返回梯度幅值和方向
gMag = hypot(gx,gy);%hypot计算三角形斜边长度的函数，在这里用于表示梯度幅值
gDir = atan2d(-gy,gx);%用于将计算图像的梯度方向

% -------------------------------------------------------------------------
% 计算HOG block的空间权重。
% -------------------------------------------------------------------------
function h = gaussianWeights(blockSize)
%--------------------------------------------------------------------------
% Matlab 的fspecial函数用法
% fspecial函数用于建立预定义的滤波算子，其语法格式为：
% h = fspecial(type)
% h = fspecial(type，para)
% 其中type指定算子的类型，para指定相应的参数；
% type参数通常可以取gaussian、average、disk、laplacian、log、prewitt
% 其中，
% gaussian为高斯低通滤波，有两个参数，hsize表示模板尺寸，默认值为【3 3】，sigma为滤波器的标准值，单位为像素，默认值为0.5.
%--------------------------------------------------------------------------
sigma = 0.5 * cast(blockSize(1), 'double');%

h = fspecial('gaussian', double(blockSize), sigma);%高斯滤波，平滑图像，图像边缘

h = cast(h, 'single');%转换数据类型为单精度浮点型

% -------------------------------------------------------------------------
% 提取有效点
% -------------------------------------------------------------------------
function validPoints = extractValidPoints(points, idx)
% 函数名：isnumeric
% 函数功能：判断输入参数是否是数字类型（包括浮点型和整型）
% 用法：t = isnumeric(A)，如果A是数字类型，返回1，否则，返回0
if isnumeric(points)
    validPoints = points(idx,:);
else    
    if isempty(coder.target)
        validPoints = points(idx);
    else
        validPoints = getIndexedObj(points, idx);
    end
end

% -------------------------------------------------------------------------
% 输入参数解析和验证
% -------------------------------------------------------------------------
function [points, isPoints, params, maxargs] = parseInputs(I, varargin)

notCodegen = isempty(coder.target);

sz = size(I);
validateImage(I);

if mod(nargin-1,2) == 1
    isPoints = true;
    points = varargin{1};
    checkPoints(points);
else
    isPoints = false;
    points = ones(0,2);
end

if notCodegen
    p = getInputParser();    
    parse(p, varargin{:});
    userInput = p.Results;          
    validate(userInput);    
    autoOverlap =  ~isempty(regexp([p.UsingDefaults{:} ''],...
        'BlockOverlap','once'));
else
    if isPoints
        [userInput, autoOverlap] = codegenParseInputs(varargin{2:end});
    else
        [userInput, autoOverlap] = codegenParseInputs(varargin{:});    
    end   
    validate(userInput);      
end

params = setParams(userInput,sz);
if autoOverlap
    params.BlockOverlap = getAutoBlockOverlap(params.BlockSize); 
end
crossValidateParams(params);

if isPoints
    maxargs = 3;
    params.WindowSize = params.BlockSize .* params.CellSize;
else
    maxargs = 3;
    params.WindowSize = params.ImageSize;
end

% -------------------------------------------------------------------------
% 输入图像验证函数
% -------------------------------------------------------------------------
function validateImage(I)
% 验证图像
validateattributes(I, {'double','single','int16','uint8','logical','gpuArray'},...
    {'nonempty','real', 'nonsparse','size', [NaN NaN NaN]},...
    'extractHOGFeatures');

sz = size(I);
coder.internal.errorIf(ndims(I)==3 && sz(3) ~= 3,...
                       'vision:dims:imageNot2DorRGB');

coder.internal.errorIf(any(sz(1:2) < 3),...
                       'vision:extractHOGFeatures:imageDimsLT3x3');

% -------------------------------------------------------------------------
% 输入参数解析的代码生成
% -------------------------------------------------------------------------
function [results, usingDefaultBlockOverlap] = codegenParseInputs(varargin)
pvPairs = struct( ...
    'CellSize',     uint32(0), ...
    'BlockSize',    uint32(0), ...
    'BlockOverlap', uint32(0),...
    'NumBins',      uint32(0),...
    'UseSignedOrientation', uint32(0));

popt = struct( ...
    'CaseSensitivity', false, ...
    'StructExpand'   , true, ...
    'PartialMatching', true);

defaults = getParamDefaults();

optarg = eml_parse_parameter_inputs(pvPairs, popt, varargin{:});

usingDefaultBlockOverlap = ~optarg.BlockOverlap;

results.CellSize  = eml_get_parameter_value(optarg.CellSize, ...
    defaults.CellSize, varargin{:});

results.BlockSize = eml_get_parameter_value(optarg.BlockSize, ...
    defaults.BlockSize, varargin{:});

results.BlockOverlap = eml_get_parameter_value(optarg.BlockOverlap, ...
    defaults.BlockOverlap, varargin{:});

results.NumBins = eml_get_parameter_value(optarg.NumBins, ...
    defaults.NumBins, varargin{:});

results.UseSignedOrientation  = eml_get_parameter_value(...
    optarg.UseSignedOrientation, ...
    defaults.UseSignedOrientation, varargin{:});

% -------------------------------------------------------------------------
% 根据块大小设置块重叠
% -------------------------------------------------------------------------
function autoBlockSize = getAutoBlockOverlap(blockSize)
szGTOne = blockSize > 1;
autoBlockSize = zeros(size(blockSize), 'like', blockSize);
autoBlockSize(szGTOne) = cast(ceil(double(blockSize(szGTOne))./2), 'like', ...
    blockSize);

% -------------------------------------------------------------------------
%参数的默认值设置函数
% -------------------------------------------------------------------------
function defaults = getParamDefaults()
intClass = 'int32';
defaults = struct('CellSize'    , cast([8 8],intClass),...
                  'BlockSize'   , cast([2 2],intClass), ...
                  'BlockOverlap', cast([1 1],intClass), ...
                  'NumBins'     , cast( 9   ,intClass), ...
                  'UseSignedOrientation', false,...
                  'ImageSize' , cast([1 1],intClass),...
                  'WindowSize', cast([1 1],intClass));
              
% -------------------------------------------------------------------------
function params = setParams(userInput,sz)
params.CellSize     = reshape(int32(userInput.CellSize), 1, 2);
params.BlockSize    = reshape(int32(userInput.BlockSize), 1 , 2);
params.BlockOverlap = reshape(int32(userInput.BlockOverlap), 1, 2);
params.NumBins      = int32(userInput.NumBins);
params.UseSignedOrientation = logical(userInput.UseSignedOrientation);
params.ImageSize  = int32(sz(1:2));
params.WindowSize = int32([1 1]);

% -------------------------------------------------------------------------
% 输入参数验证函数
% -------------------------------------------------------------------------
function validate(params)

checkSize(params.CellSize,  'CellSize');

checkSize(params.BlockSize, 'BlockSize');

checkOverlap(params.BlockOverlap);

checkNumBins(params.NumBins);

checkUsedSigned(params.UseSignedOrientation);

% -------------------------------------------------------------------------
%交叉验证输入值
% -------------------------------------------------------------------------
function crossValidateParams(params)
% 交叉验证输入参数
                   
coder.internal.errorIf(any(params.BlockOverlap(:) >= params.BlockSize(:)), ...
    'vision:extractHOGFeatures:blockOverlapGEBlockSize');

% -------------------------------------------------------------------------
function parser = getInputParser()
persistent p;
if isempty(p)
    
    defaults = getParamDefaults();
    p = inputParser();
    
    addOptional(p, 'Points', []);    
    addParameter(p, 'CellSize',     defaults.CellSize);
    addParameter(p, 'BlockSize',    defaults.BlockSize);
    addParameter(p, 'BlockOverlap', defaults.BlockOverlap);
    addParameter(p, 'NumBins',      defaults.NumBins);
    addParameter(p, 'UseSignedOrientation', defaults.UseSignedOrientation);    
    
    parser = p;
else
    parser = p;
end

% -------------------------------------------------------------------------
function checkPoints(pts)

if vision.internal.inputValidation.isValidPointObj(pts)    
    vision.internal.inputValidation.checkPoints(pts, mfilename, 'POINTS');   
else
    validateattributes(pts, ...
        {'int16', 'uint16', 'int32', 'uint32', 'single', 'double'}, ...
        {'2d', 'nonsparse', 'real', 'size', [NaN 2]},...
        mfilename, 'POINTS');
end

% -------------------------------------------------------------------------
function checkSize(sz,name)

vision.internal.errorIfNotFixedSize(sz, name);
validateattributes(sz, {'numeric'}, ...
                   {'real','finite','positive','nonsparse','numel',2,'integer'},...
                   'extractHOGFeatures',name); 

% -------------------------------------------------------------------------
function checkOverlap(sz)

vision.internal.errorIfNotFixedSize(sz, 'BlockOverlap');
validateattributes(sz, {'numeric'}, ...
                   {'real','finite','nonnegative','nonsparse','numel',2,'integer'},...
                   'extractHOGFeatures','BlockOverlap');

% -------------------------------------------------------------------------
function checkNumBins(x)

vision.internal.errorIfNotFixedSize(x, 'NumBins');
validateattributes(x, {'numeric'}, ...
                   {'real','positive','scalar','finite','nonsparse','integer'},...
                   'extractHOGFeatures','NumBins');

% -------------------------------------------------------------------------
function checkUsedSigned(isSigned)

vision.internal.errorIfNotFixedSize(isSigned, 'UseSignedOrientation');
validateattributes(isSigned, {'logical','numeric'},...
    {'nonnan', 'scalar', 'real','nonsparse'},...
    'extractHOGFeatures','UseSignedOrientation');

% -------------------------------------------------------------------------
function checkNumOutputsForCodegen(numOut, maxargs)

if ~isempty(coder.target)
    % 如果生成代码，则不允许HOG可视化
    coder.internal.errorIf(numOut > maxargs-1,...
        'vision:extractHOGFeatures:hogVisualizationNotSupported');    
end
