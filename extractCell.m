function [hogfeateur,features, varargout] = extractCell(I,varargin)
%extractHOGFeatures��ȡHOG������
%features = extractHOGFeatures��I�������ɫ��Ҷ�ͼ��I����ȡHOG����������1��N�������������� 
%��Щ������ͼ��������ľֲ���״��Ϣ���б��룬������������񣬰������࣬���͸��٣�ͼ������������
%
%HOG��������N����ͼ���С�������г��Ĳ���ֵ��
%�ο� <a href="matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'extractHOGFeatures')" >documentation</a> for more information. 
%
%[features, validPoints] = extractHOGFeatures(I, points) ������ͼ��I�еĵ�λ����Χ��ȡ��HOG������
%�ú���������validPoints�����а��������λ�ã�����Χ��[CellSize.* BlockSize]������ȫ������ͼ��I�С� 
%���Խ������ָ��Ϊ[xy]���꣬SURFPoints��cornerPoints��MSERRegions��BRISKPoints��M��2���� 
%����Щ��������κα�������Ϣ���������ԡ� validPoints��������������ͬ��.
%
%  [..., visualization] = extractHOGFeatures(I, ...) ���ؿ���ʹ��plot��visualization����ʾ��HOG����������HOG�����Ŀ��ӻ���
%
%  [...] = extractHOGFeatures(..., Name, Value) ָ����������/ֵ�ԣ�����������
%
%  'CellSize'     ��2Ԫ������������ָ��HOG��Ԫ�Ĵ�С��������Ϊ��λ���� 
%                 ѡ��ϴ��cell�ߴ��Բ�����ģ�Ŀռ���Ϣ����Ҫ������С��ϸ��Ϊ���ۡ�
%                 
%                 Default: [5 8]
%
%  'BlockSize'    ��2Ԫ������������ָ�����е�cell�������� 
%                 ��Ŀ��Сֵ�ή��ʹ�ֲ������仯��С����������
%
%                 Default: [2 2]
%
%  'BlockOverlap' ��2Ԫ������������ָ�����ڿ�֮����ص�cell������ 
%                 ѡ������Ϊ���Сһ����ص����֣���ȷ���㹻�ĶԱȶȹ�һ���� 
%                 �ϴ���ص�ֵ���������ӵ�����������СΪ���۲��������Ϣ�� 
%                 ��ȡ��λ����Χ��HOGҪ��ʱ����������Ч��
% 
%                 Default: ceil(BlockSize/2)
%                  
%  'NumBins'      һ�����ı�����ָ������ֱ��ͼ�е�bin���� ���Ӵ�ֵ�ɱ������ϸ�ķ���ϸ�ڡ�
%                 
%                 Default: 9
%
%'UseSignedOrientation'��һ���߼������� ���Ϊtrue���򽫷���ֵ�ϲ���-180��180��֮��ľ��ȼ���ĺϲ�bins�С� 
%                      ���򣬽�����ֵ�ϲ���0��180֮�䣬���н�С��0��thetaֵ����theta + 180 bin�С� 
%                      ʹ�ô����ŵķ�����԰�������ͼ�������е����������밵�����ɡ�
%
%
%     Default: false
%
% Class Support
% -------------
%�����ͼ���ҿ�����uint8��int16��double��single���߼�����������ʵ���ͷ�ϡ��ġ�
%POINTS������SURFPoints��cornerPoints��MSERRegions��BRISKPoints��int16��uint16��int32��uint32������˫��
%
%
% ���ú�����ʾ��1-��ͼ������ȡHOG������
% -----------------------------------------------
%
%    I1 = imread('gantrycrane.png');
%    [hog1, visualization] = extractHOGFeatures(I1,'CellSize',[32 32]);
%    subplot(1,2,1);
%    imshow(I1);
%    subplot(1,2,2);
%    plot(visualization);
%
% ���ú�����ʾ��2-��ȡ�սǵ㸽����HOG������
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
% �����Ĳ��֣���μ�extractFeatures��extractLBPFeatures��detectHarrisFeatures��
%detectFASTFeature��detectMinEigenFeatures��detectSURFFeature��
%detectMSERFeatures��detectBRISKFeatures


%#codegen
%#ok<*EMCA>

notCodegen = isempty(coder.target);%�жϴ�������Ŀ���Ƿ�Ϊָ��Ŀ��

[points, isPoints, params, maxargs] = parseInputs(I,varargin{:});

% ������������outputs numbers��
if notCodegen
    nargoutchk(0,maxargs);%������
else    
    checkNumOutputsForCodegen(nargout, maxargs);%������ɴ��룬������HOG���ӻ�
end

if isPoints
    %�������������͵�������ȡ����һ���������ӵ���ȡ�ʹ�����ͼ����ȡ
    [features, validPoints] = extractHOGFromPoints(I, points, params);
    
    if nargout >= 2%����ú�����������ĸ���
        varargout{1} = validPoints;
    end
    
    %nargout����������matlab�ж���һ������ʱ�� 
    %�ں������ڲ��� nargoutָ������������ĸ�����narginָ������������ĸ������� �ر����������˿ɱ�����б�ĺ����У� ��nargout��ȡ������������ܷ��㡣
    
    if notCodegen
        if nargout == 3
            params.Points = validPoints;
            varargout{2}  = vision.internal.hog.Visualization(features, params);
        end
    end
else   
   
    [hogfeateur,features] = extractHOGFromImage(I, params);  
   
    if notCodegen
        if nargout == 3%ԭ����2
            
            varargout{1} = vision.internal.hog.Visualization(features, params);
            
        end
    end
end
 
% -------------------------------------------------------------------------
% ������ͼ������ȡHOG���� 
% -------------------------------------------------------------------------
function [hogfeateur,features] = extractHOGFromImage(I, params)
[gMag, gDir] = hogGradient(I);%����ͼ���ݶȷ�ֵ���ݶȽǶȣ�����������386�и���

[gaussian, spatial] = computeWeights(params);%�����˹�Ϳռ�Ȩ�أ��ú�����375�и�����

[hogfeateur, features] = extractHOG(gMag, gDir, gaussian, spatial, params);

% -------------------------------------------------------------------------
% ���ݵ�λ����ȡHOG����
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
    
    % ROI �Ե�λ��Ϊ����
    roi(1:2) = cast(round(xy(i,[2 1])), featureClass) - halfSize;
    
    % ����ͼ������ȫ����ROIʱ����
    if all(roi(1:2) >= 1) && ...
            roi(1)+roi(3)-1 <= params.ImageSize(1) && ...
            roi(2)+roi(4)-1 <= params.ImageSize(2)
        
        validPointCount = validPointCount + 1;
        
        [gMag, gDir] = hogGradient(I, roi);
               
        hog = extractHOG(gMag, gDir, gaussian, spatial, params);
        
        features(validPointCount,:) = hog(:);
        validPointIdx(validPointCount) = i; % �洢��Ч����
    end
    
end

features = features(1:validPointCount,:);

validPoints = extractValidPoints(points, validPointIdx(1:validPointCount));

% -------------------------------------------------------------------------
% �ڸ����ݶȷ�ֵ�ͷ�����������ȡHOG����
% -------------------------------------------------------------------------
function [cellhog,hog] = extractHOG(gMag, gDir, gaussianWeights, weights, params)
    %----------------------------------------------------------------------
    %
    %���������gMag��gDir������ͼ��img��ͬ�ߴ�Ĵ�С������ÿ�����ݷֱ��Ǹ�ͼ��ÿ�����ص��ݶȷ�ֵ���ݶȷ�������Ǹ�����
    %
    %----------------------------------------------------------------------
% if isempty(coder.target)         
%     hog = visionExtractHOGFeatures(gMag, gDir, gaussianWeights, params, weights); 
%     cellhog = rand(2,3);
% else
    featureClass = 'single';%������������Ϊ������
    
    if params.UseSignedOrientation%Ĭ��Ϊ0����Ϊ����ϰ����ʹ��180��
        % ����gDir�ķ�Χ�� [0 360]֮��
        histRange = single(360);%�������ͱ���
    else
        % �ѷ���ת�����޷����ͣ���Χ�� [0 180]֮��
        histRange = single(180);%�������ͱ���
    end
    
    %��� gDir ��Χ�� [-180 180], ת����  [0 180] ���� [0 360]֮��
    negDir = gDir < 0;%��Ϊ1����Ϊ0
    gDir(negDir) = histRange + gDir(negDir);
    
    % Ϊ���е�cells��λ����bin
    binWidth = histRange/cast(params.NumBins, featureClass);%���磬������180�ȵ�����£���9��bin����ÿ��bin�Ŀ����20�ȡ�
    [x1, b1] = computeLowerHistBin(gDir, binWidth);
    %���øù�ʽ�������ǰ���ص��ݶȷ�����bin���ĸ�λ�á�x1��ʾ��ǰ�ݶȷ���Ƕ������Ƕ��м�ֵ��
    %�����һ��bin��0-20�ȣ���ô���ֵ����10��b1��ʾ�ýǶ���bin�еĵڼ���
    wDir = 1 - (gDir - x1)./binWidth;%����Ȩ�أ����м�ֵƫ��Խ��Ȩ��ԽС�����Ǻ���Ȼ�ġ������Ļ������Լ�����������ԡ�����һ����ͼ���С��ͬ�ľ���
    
    blockSizeInPixels = params.CellSize.*params.BlockSize;%��8��8��*��2��2������ʾblock�������ص�������16��16����������64
    blockStepInPixels = params.CellSize.*(params.BlockSize - params.BlockOverlap);%��ʾblockÿ���ƶ��������ء�
    
    r = 1:blockSizeInPixels(1);%��1��2��3��4��....16
    c = 1:blockSizeInPixels(2);%��1��2��3��4��....16
    
    nCells  = params.BlockSize;%ÿ��block�ĳߴ磬���ڼ���ÿ��block��cell�ĸ��� 2*2
    nBlocks = vision.internal.hog.getNumBlocksPerWindow(params);%ÿ��������block������
    size(nBlocks)
    numCellsPerBlock = nCells(1)*nCells(2);%ÿ��block��cell�ĸ���,4
    
    %���ȼ���ÿ��block���ж��ٸ�bin
    hog = coder.nullcopy(zeros([params.NumBins*numCellsPerBlock, nBlocks],featureClass)); %��bin����������ÿ��block��cell��������������block����������
    %hog�Ĵ洢˼���ǣ�ÿ��binռ�Ÿ���λ��ÿ��cellһ��bin��ÿ��block��4��cell����ÿ��һ��block��bin
    %��9*4,n����n��ÿ��������block�ĸ�����ÿ��cell����9��bin��ÿ��block����4��cell������ÿ��block����36��b�ռ䣬�ֱ���4��cell��bin
    %nBlocks = 15*7
    %��������hog����˼����ǣ�ÿһ�о���һ��block��������bin��������1��9��cell��1��1����bin��10��18��cell(1,2)��bin��19��27��cell(2,1)��bin��28��36��cell(2,2)��bin
    % ɨ�贰���ڵ�����block���𵽱���ͼ���Ŀ��
    for j = 1:nBlocks(2)%;��       
        for i = 1:nBlocks(1)%��            
            %�Է���Ȩ�ؽ������Բ�ֵ��Ȩ
            wz1 = wDir(r,c); %��ʵ����  wz1 =  wDir.��ʾ��ǰ�ݶȷ������ڸ�bin��Ȩ�ء�����һ����block��С��ͬ�ľ���
            w = trilinearWeights(wz1, weights); %                       
            % �Է�ֵ���и�˹�˲���Ȩ
            m = gMag(r,c) .* gaussianWeights; %�����ط�ֵ���и�˹ƽ����������һ����block��С��ͬ�ľ���           
            % interpolate magnitudes for binning ���ݶȷ�ֵ���в�ֵ���浽bin��
            
            %����Ĳ�������˽��ݶȷ�ֵ���ݶȷ�����м�Ȩ�ںϵ����񣬴Ӷ����������������ϳ�һ�����ݣ�
            mx1y1z1 = m .* w.x1_y1_z1;
            mx1y1z2 = m .* w.x1_y1_z2;
            mx1y2z1 = m .* w.x1_y2_z1;
            mx1y2z2 = m .* w.x1_y2_z2;
            mx2y1z1 = m .* w.x2_y1_z1;
            mx2y1z2 = m .* w.x2_y1_z2;
            mx2y2z1 = m .* w.x2_y2_z1;
            mx2y2z2 = m .* w.x2_y2_z2;            
            %--------------------------------------------------------------
            %���濪ʼ����ô洢���ĸ�bin����
            %--------------------------------------------------------------
            orientationBins = b1(r,c);%�����صĽǶ��б�����ȡĳ�����صĽǶȶ�Ӧ�������            
            % ����ֱ��ͼ��ʼ��Ϊ��
            h = zeros(params.NumBins+2, nCells(1)+2, nCells(2)+2, featureClass);%�����ȣ�h���ڴ洢ÿ��block�е�bin��
            %--------------------------------------------------------------
            %�����Ҫ���ظ㶮ֱ��ͼ�Ĵ洢����
            %h�������һ����10��11*10����ά������������Ϊ���������ͣ����е�����10��ʾ�Ƕ�����λ�ã�1��2��3��4��5��6��7��8��9��10
            %1��ʾ��0-20�ȣ�2��ʾ��20-40�ȣ��Դ�����.........
            %���У�ÿ��11*10�ľ������ڴ洢
            %--------------------------------------------------------------            
            % ����Ҳ�����ж�ȡblock�е�ÿ�����ء�
            for x = 1:blockSizeInPixels(2)%1��16,����������ʾ�����ߣ�n
                cx = weights.cellX(x);%�ж�������������ĸ�cell
                for y = 1:blockSizeInPixels(1)%1��16
                    z  = orientationBins(y,x);%������ص��ݶȷ��������ĸ�bin�������жϳ����ĽǶ���ʵ�����ڵĴ����ڣ���z+1�Ǳ����z�����һ������
                    %��Ϊ�㷨��Ҫ����������bin�ڽ��м�Ȩ�����ݾ�����м�Ȩ��
                    cy = weights.cellY(y);%1��2��3
                    
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
            
            % hװ�����������������Ϊ���洴��hʱ��x��y��+2��֮���Լ�2������Ϊ�����ڼ������������ĸ�cellʱ������1��2��3��Ȼ��3����������Ҫ�ġ�
            h(2,:,:)     = h(2,:,:)     + h(end,:,:);
            h(end-1,:,:) = h(end-1,:,:) + h(1,:,:);
            
            % ��������ֱ��ͼ����Ч����
            h = h(2:end-1,2:end-1,2:end-1);
            
            %���򻯲�������ӵ���������         
            hog(:,i,j) = normalizeL2Hys(h(:));%�Ӵ����Ϲ۲죬ÿ��h��ʾһ��block�е�����bin����hog�У�ÿ��j��ʾһҳ��ÿһҳ�ľ������һ��block������binֵ��       
            r = r + blockStepInPixels(1);
        end
        r = 1:blockSizeInPixels(1);
        c = c + blockStepInPixels(2);
    end
    cellhog = hog;
    hog = reshape(hog, 1, []);%��hog������reshapeһ��һ��N�е����������ڵ�SVM���㷨��ѵ��
%end

% -------------------------------------------------------------------------
% ʹ��L2-Hys��һ������
% -------------------------------------------------------------------------
function x = normalizeL2Hys(x)
classToUse = class(x);
x = x./(norm(x,2) + eps(classToUse)); % L2 norm
x(x > 0.2) = 0.2;                     % ����0.2�Ķ��޼��� 0.2
x = x./(norm(x,2) + eps(classToUse)); % �ظ�L2����

% -------------------------------------------------------------------------
% ����cell�Ͽռ�ֱ��ͼ�Ĳ�ֵȨ�أ���ʵ����˫���Բ�ֵ
%˫���Բ�ֵ��˼���ǣ��ֱ�������㣨x��y����Χ�ĸ����Ȩ�أ��Ϳ��Ը�����ЩȨ�ؼ�������������
% -------------------------------------------------------------------------
function weights = spatialHistWeights(params)
%--------------------------------------------------------------------------
%�����������Ҫ�����Ƕ�ͼ���ϵĵ����˫���Կռ��ֵ�����ÿ������֮��Ĳ�ֵȨ�أ�
%��Ҫ�Ƕ��ĸ����м��������й��ƣ�
%���Ƚ���x�᷽��Ĳ�ֵ��Ȼ�����Y�᷽��Ĳ�ֵ������i���Բ�ֵ�Ľ�����ֵ˳���޹أ��ȼ����Ǹ�����������ν��
%
%��Ҫָ�����ǣ�Ϊ�˼��ٻ����ѡƱ�ڷ����λ���϶��������ڵ�bins����֮�����˫�����ڲ塣
%�����Ըú�����ʹ�ã��ֱ��ڼ��������ݶȷ��������λ�õ�ʱ����в�ֵ
%--------------------------------------------------------------------------

%�ռ�ֱ��ͼȨ��
% ��ԣ�x��y����Χ��4�������2D��ֵȨ��
%
% (x1,y1) o---------o (x2,y1)
%         |         |
%         |  (x,y)  |
%         |         |
% (x1,y2) o---------o (x2,y2)
%
% ��x��y����HOG��block�ڵ���������
%
% (x1,y1); (x2,y1); (x1,y2); (x2,y2) ��һ�����ڵ�cell����

%���㵱ǰblock���ڵ�λ�ø��ǵ�ͼ������ķ�Χ
width  = single(params.BlockSize(2)*params.CellSize(2));%��Ŀ����cell�Ŀ�
height = single(params.BlockSize(1)*params.CellSize(1));%��ĸ߳���cell�ĸߣ�

x = 0.5:1:width;
y = 0.5:1:height;

[x1, cellX1] = computeLowerHistBin(x, params.CellSize(2));%function [x1, b1] = computeLowerHistBin(x, binWidth)
[y1, cellY1] = computeLowerHistBin(y, params.CellSize(1));

wx1 = 1 - (x - x1)./single(params.CellSize(2));%8
wy1 = 1 - (y - y1)./single(params.CellSize(1));

%ʹ�ýṹ�壬����˫���Բ�ֵ�ļ���ʽ��������ʽ��Ĭ�����ĸ���֪������ֱ�Ϊ (0, 0)��(0, 1)��(1, 0) �� (1, 1)������µ������������ô��ֵ��ʽ�Ϳ��Ի���Ϊ
%�ο���ҳhttps://blog.csdn.net/xjz18298268521/article/details/51220576
weights.x1y1 = wy1' * wx1;
weights.x2y1 = wy1' * (1-wx1);
weights.x1y2 = (1-wy1)' * wx1;
weights.x2y2 = (1-wy1)' * (1-wx1);

% also store the cell indices  Ҳ�Ǵ洢cell����
weights.cellX = cellX1;
weights.cellY = cellY1;

% -------------------------------------------------------------------------
% ������������Ȩ�أ�ȡ���ǿռ��еİ˸��㣬��������˸���֮������һ�����λ��
% -------------------------------------------------------------------------
function weights = trilinearWeights(wz1, spatialWeights)

% ��ʹ��ǰ����ṹ�ֶ�
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
% ����С�ڻ����x��,��ӽ�bin���ĵ�x1
% -------------------------------------------------------------------------
function [x1, b1] = computeLowerHistBin(x, binWidth)
% Bin ������
width    = single(binWidth);%ת���ɵ�����
invWidth = 1./width;%����
bin      = floor(x.*invWidth - 0.5);%floor���������������ȡ��������-1.2��ȡֵ-2��
% Bin ������ x1
x1 = width * (bin + 0.5);
% ��2�Ի�û���1������
b1 = int32(bin + 2);

% -------------------------------------------------------------------------
%�����˹�Ϳռ�Ȩ��
% -------------------------------------------------------------------------
function [gaussian, spatial] = computeWeights(params)
blockSizeInPixels = params.CellSize.*params.BlockSize;%
gaussian = gaussianWeights(blockSizeInPixels);%��˹�˲���ƽ��ͼ��͹�Ա�Ե
spatial  = spatialHistWeights(params);

% -------------------------------------------------------------------------
% ʹ�ò���˲���[-1 0 1]�����ݶȼ��㡣 ʹ��ǰ������ͼ��߽紦�Ľ��䡣 
% ����X����ʱ��������ݶȷ�����-180��180��֮�䡣
% -------------------------------------------------------------------------
function [gMag, gDir] = hogGradient(img,roi)

if nargin == 1  %nargin �������ж�������������ĺ�����������������ж��Ƿ�ֻ��ͼ���ROI���������HOG������
    roi = [];    
    imsize = size(img);
else
    imsize = roi(3:4);
end

img = single(img);%ͼ������ת���ɵ�����

if ndims(img)==3%���ͼ��ʱ��ά��ɫͼ��
    rgbMag = zeros([imsize(1:2) 3], 'like', img);
    rgbDir = zeros([imsize(1:2) 3], 'like', img);
    
    for i = 1:3
        %
        [rgbMag(:,:,i), rgbDir(:,:,i)] = computeGradient(img(:,:,i),roi);
    end
    
    % �ҵ�ÿ�����ص������ɫ����
    [gMag, maxChannelIdx] = max(rgbMag,[],3);
    
    % �ӷ�������λ����ȡ�ݶȷ���
    sz = size(rgbMag);
    [rIdx, cIdx] = ndgrid(1:sz(1), 1:sz(2));
    ind  = sub2ind(sz, rIdx(:), cIdx(:), maxChannelIdx(:));
    gDir = reshape(rgbDir(ind), sz(1:2));
else
    [gMag,gDir] = computeGradient(img,roi);
end

% -------------------------------------------------------------------------
% �ú�������������TOI������ͼ���ݶ�Ix��Iy�ļ��㷽����������ĺ����л�ֱ�ӵ��á�
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

% ���ROI��ͼ��߿��ϣ����Ʊ߿�����. 
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
%�ú��������ȸ���������ͼ����ݶȼ��㷽����Ȼ��ʹ���������computeGradientROI����������Ҫ����ROI����ʱ���ݶȡ����������ݶȷ�ֵ���ݶȷ���ļ��㷽��
%
% -------------------------------------------------------------------------
function [gMag,gDir] = computeGradient(img,roi)
operator_gx = [-1,0,1] ;
operator_gy = operator_gx' ;
if isempty(roi)
    %������ü���ROI
    %gx��gy���������ͼ����x�᷽���y�᷽���ϵ��ݶȵ�
    gx = zeros(size(img), 'like', img);%������ͼ��img�����ľ���
    gy = zeros(size(img), 'like', img);%������ͼ��img�����ľ���
    
    gx(:,2:end-1) = conv2(img, operator_gx, 'valid');%����˲���img�Ǳ������ͼ��[-1,0,1]�Ǿ���ˣ���ʵ�������ݶ�
    gy(2:end-1,:) = conv2(img, operator_gy, 'valid');
    
    % �߽��ϵ��ݶ�
    gx(:,1)   = img(:,2)   - img(:,1);
    gx(:,end) = img(:,end) - img(:,end-1);
    gy(1,:)   = img(2,:)   - img(1,:);
    gy(end,:) = img(end,:) - img(end-1,:);
else
    %����ROI����Ҫʹ��ר�ŵ��ݶȼ��㺯��
    [gx, gy] = computeGradientROI(img, roi);
end

% �����ݶȷ�ֵ�ͷ���
gMag = hypot(gx,gy);%hypot����������б�߳��ȵĺ��������������ڱ�ʾ�ݶȷ�ֵ
gDir = atan2d(-gy,gx);%���ڽ�����ͼ����ݶȷ���

% -------------------------------------------------------------------------
% ����HOG block�Ŀռ�Ȩ�ء�
% -------------------------------------------------------------------------
function h = gaussianWeights(blockSize)
%--------------------------------------------------------------------------
% Matlab ��fspecial�����÷�
% fspecial�������ڽ���Ԥ������˲����ӣ����﷨��ʽΪ��
% h = fspecial(type)
% h = fspecial(type��para)
% ����typeָ�����ӵ����ͣ�paraָ����Ӧ�Ĳ�����
% type����ͨ������ȡgaussian��average��disk��laplacian��log��prewitt
% ���У�
% gaussianΪ��˹��ͨ�˲���������������hsize��ʾģ��ߴ磬Ĭ��ֵΪ��3 3����sigmaΪ�˲����ı�׼ֵ����λΪ���أ�Ĭ��ֵΪ0.5.
%--------------------------------------------------------------------------
sigma = 0.5 * cast(blockSize(1), 'double');%

h = fspecial('gaussian', double(blockSize), sigma);%��˹�˲���ƽ��ͼ��ͼ���Ե

h = cast(h, 'single');%ת����������Ϊ�����ȸ�����

% -------------------------------------------------------------------------
% ��ȡ��Ч��
% -------------------------------------------------------------------------
function validPoints = extractValidPoints(points, idx)
% ��������isnumeric
% �������ܣ��ж���������Ƿ����������ͣ����������ͺ����ͣ�
% �÷���t = isnumeric(A)�����A���������ͣ�����1�����򣬷���0
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
% ���������������֤
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
% ����ͼ����֤����
% -------------------------------------------------------------------------
function validateImage(I)
% ��֤ͼ��
validateattributes(I, {'double','single','int16','uint8','logical','gpuArray'},...
    {'nonempty','real', 'nonsparse','size', [NaN NaN NaN]},...
    'extractHOGFeatures');

sz = size(I);
coder.internal.errorIf(ndims(I)==3 && sz(3) ~= 3,...
                       'vision:dims:imageNot2DorRGB');

coder.internal.errorIf(any(sz(1:2) < 3),...
                       'vision:extractHOGFeatures:imageDimsLT3x3');

% -------------------------------------------------------------------------
% ������������Ĵ�������
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
% ���ݿ��С���ÿ��ص�
% -------------------------------------------------------------------------
function autoBlockSize = getAutoBlockOverlap(blockSize)
szGTOne = blockSize > 1;
autoBlockSize = zeros(size(blockSize), 'like', blockSize);
autoBlockSize(szGTOne) = cast(ceil(double(blockSize(szGTOne))./2), 'like', ...
    blockSize);

% -------------------------------------------------------------------------
%������Ĭ��ֵ���ú���
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
% ���������֤����
% -------------------------------------------------------------------------
function validate(params)

checkSize(params.CellSize,  'CellSize');

checkSize(params.BlockSize, 'BlockSize');

checkOverlap(params.BlockOverlap);

checkNumBins(params.NumBins);

checkUsedSigned(params.UseSignedOrientation);

% -------------------------------------------------------------------------
%������֤����ֵ
% -------------------------------------------------------------------------
function crossValidateParams(params)
% ������֤�������
                   
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
    % ������ɴ��룬������HOG���ӻ�
    coder.internal.errorIf(numOut > maxargs-1,...
        'vision:extractHOGFeatures:hogVisualizationNotSupported');    
end
