function [ Issnew ] = getSegmentationColors( I,MaskSemantic )
% THIS FUNCTION RETURNS A COLOR IMAGE. THIS IMAGE IS A MASK SEGMENTATION 
% IMAGE WHERE EACH REGION: PLAQUE AND BACKGROUND IS SHOWN INF DIFFERENT
% COLORS
%
% INPUT
% I: THE ORIGINAL IMAGE
% MaskSemantic: THE MASK SEGMENTATION 
%
% OUTPUT
% Issnew: THE COLOR IMAGE WITH THE MASK SEGMENTATION

%  figure();imshow(MaskSemantic);

    %Find the pixels from each component after lines are removed
    Issnew=I;
    [y1new,x1new]=find(MaskSemantic==0);
    [y2new,x2new]=find(MaskSemantic==1);


    %RESCRIURE NOMS ETIQUETES!!!
    
    %Print each region with a color
    [Issnew]=printSegment(Issnew,x1new,y1new,0,250,0);%Near Wall
    [Issnew]=printSegment(Issnew,x2new,y2new,250,0,0);%Lumen-Left

% figure();imshow(Issnew);

end

