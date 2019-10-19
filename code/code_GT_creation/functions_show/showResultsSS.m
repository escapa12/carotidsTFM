
function showResultsSS(stripI,stripSS,size_x,size_y,nimages,len,direction)
% This function shows the mosaic for the mask segmentaiton results.
% It shows each pair of images: The original image and its colored segmantation
%
%---INPUTS---
%stripI: contains all the original images in a strip
%stripSS: contains all the images with the masks segmentation result in a strip
%stripGTImages: contains all the GT masks in a strip
%nimages: The number of pair images that will be shown
%size_x, size_y: image dimensions
%len: the number of pair images shown in eahc mosaic


% figure(); imshow(stripSS);
     
    nmosaics=fix(nimages/len);%Set the number of mosaics

for m=1:nmosaics
    
    mosaic=[];%Initialize the image of the mosaic to be shown
    
    %Chose a sub-strip image in each strip
    stripIaux=stripI(:,(m-1)*len*size_x+1:m*len*size_x,:);
    stripSSaux=stripSS(:,(m-1)*len*size_x+1:m*len*size_x,:);
    
    %Built te strip image for the mosaic: The original images
    auxMosaicOri = [];
    for j=1:len
        %Choose one image and add it in the mosaic
        if direction==1
            auxMosaicOri=[auxMosaicOri stripIaux(1:size_y,(size_x*(j-1)+1):size_x*j,:)];
        else                    
            auxMosaicOri=[auxMosaicOri;stripIaux(1:size_y,(size_x*(j-1)+1):size_x*j,:)];
        end;
    end

    %Built te strip image for the mosaic: The Mask segmentation images
    auxMosaicSS = [];
    for j=1:len
        %Choose one image and add it in the mosaic
         if direction==1
            auxMosaicSS=[auxMosaicSS stripSSaux(1:size_y,(size_x*(j-1)+1):size_x*j,:)];
         else
            auxMosaicSS=[auxMosaicSS; stripSSaux(1:size_y,(size_x*(j-1)+1):size_x*j,:)];
         end;
    end

    %Built the horitzontal (or veritcal) mosaic joinning both strips
    if direction==1
         mosaic=[auxMosaicOri;auxMosaicSS];
    else
         mosaic=[auxMosaicOri auxMosaicSS];
    end;

    %Show the mosaic
    figure(); imshow(mosaic);

end;   
        
