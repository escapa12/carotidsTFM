function showMosaicSemSegm(stripImages,stripMaskResult,size_x,size_y,len)
%Qualitative validation
%
%---INPUTS---
%stripImages: contains all the images in a strip
%stripMaskResult: contains all the masks with the results of the post-process step in a strip
%stripGTImages: contains all the GT masks in a strip
%size_x, size_y: image dimensions
%len: total number of images

%GLOBAL VARIABLES

if (len==48)
    %48 IMAGES, 4 mosaics  in 3x4
    nMOSAICS=4; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=12; %set the number of images per mosaic
    rowMosaic=4;%number of images per row in a mosaic
    colMosaic=3;%number of iamges per column in a mosaic
    
    
elseif(len==36)
    %36 IMAGES, 4 mosaics in 3x3
    nMOSAICS=4; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=9; %set the number of images per mosaic
    rowMosaic=3;%number of images per row in a mosaic
    colMosaic=3;%number of iamges per column in a mosaic  
    
elseif(len==24)
    %24 IMAGES, 4 mosaics in 2x3
    % nMOSAICS=4; %Number of mosaics (each mosaic one figure)
    % imgsMOSAIC=6; %set the number of images per mosaic
    % rowMosaic=3;%number of images per row in a mosaic
    % colMosaic=2;%number of iamges per column in a mosaic

    %24 IMAGES, 2 mosaics in 3x4
    nMOSAICS=2; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=12; %set the number of images per mosaic
    rowMosaic=4;%number of images per row in a mosaic
    colMosaic=3;%number of iamges per column in a mosaic

elseif(len==20)
    %20 IMAGES, 2 mosaics in 2x5
    nMOSAICS=2; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=10; %set the number of images per mosaic
    rowMosaic=5;%number of images per row in a mosaic
    colMosaic=2;%number of iamges per column in a mosaic

elseif(len==18)
    %18 IMAGES, 2 mosaics in 3x3
    nMOSAICS=2; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=9; %set the number of images per mosaic
    rowMosaic=3;%number of images per row in a mosaic
    colMosaic=3;%number of iamges per column in a mosaic
    
elseif(len==12)   
    %12 IMAGES, 1 mosaic in 3x4
    nMOSAICS=1; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=12; %set the number of images per mosaic
    rowMosaic=4;%number of images per row in a mosaic
    colMosaic=3;%number of iamges per column in a mosaic
    
elseif(len==10)   
    %10 IMAGES, 1 mosaic in 2x5
    nMOSAICS=1; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=10; %set the number of images per mosaic
    rowMosaic=5;%number of images per row in a mosaic
    colMosaic=2;%number of iamges per column in a mosaic    
elseif(len==9)
    %9 IMAGES, 1 mosaic in 3x3
    nMOSAICS=1; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=9; %set the number of images per mosaic
    rowMosaic=3;%number of images per row in a mosaic
    colMosaic=3;%number of iamges per column in a mosaic

elseif(len==6)
    %6 IMAGES, 1 mosaic in 2x3
    nMOSAICS=1; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=6; %set the number of images per mosaic
    rowMosaic=3;%number of images per row in a mosaic
    colMosaic=2;%number of iamges per column in a mosaic
    
elseif(len==8)
    %8 IMAGES, 1 mosaic in 2x4
    nMOSAICS=1; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=8; %set the number of images per mosaic
    rowMosaic=4;%number of images per row in a mosaic
    colMosaic=2;%number of iamges per column in a mosaic
    
   
elseif(len==5)
    %5 IMAGES, 5 mosaics in 1x1
    nMOSAICS=5; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=1; %set the number of images per mosaic
    rowMosaic=1;%number of images per row in a mosaic
    colMosaic=1;%number of iamges per column in a mosaic
    
elseif(len==4)
    %4 IMAGES, 2 mosaics in 1x2
    nMOSAICS=1; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=4; %set the number of images per mosaic
    rowMosaic=2;%number of images per row in a mosaic
    colMosaic=2;%number of iamges per column in a mosaic
    
elseif(len==3)
    %3 IMAGES, 3 mosaics 1x1
    nMOSAICS=3; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=1; %set the number of images per mosaic
    rowMosaic=1;%number of images per row in a mosaic
    colMosaic=1;%number of iamges per column in a mosaic 
    
elseif(len==2)
    %1 IMAGES, 1 mosaics 1x2
    nMOSAICS=1; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=2; %set the number of images per mosaic
    rowMosaic=2;%number of images per row in a mosaic
    colMosaic=1;%number of iamges per column in a mosaic
      
elseif(len==1)
    %1 IMAGES, 1 mosaic in 1x1
    nMOSAICS=1; %Number of mosaics (each mosaic one figure)
    imgsMOSAIC=1; %set the number of images per mosaic
    rowMosaic=1;%number of images per row in a mosaic
    colMosaic=1;%number of iamges per column in a mosaic
    
end;

    %Order of the images in the mosaic
    ordre = (1:len);

    %Construct nMOSAICS with imgsMOSAIC images
    for t=1:nMOSAICS
        
        mosaic=[];
%         ncol=size_x*rowMosaic;

        %--Built the mosaic of the images
        for k=1:colMosaic
            auxMosaic = [];
            for i=1:rowMosaic
                j=(k-1)*rowMosaic + i;
                j=ordre(j);
                auxMosaic=[auxMosaic stripImages(1:size_y,(size_x*(j-1)+1):size_x*j)];
            end
             mosaic=[mosaic; auxMosaic];
        end;
       
        
        figure(); imshow(mosaic);
    
        %Show figrues inside 1 mosaic
        for i=1:imgsMOSAIC
            k=ordre(i);
            maskPostProcess=stripMaskResult(1:size_y,(size_x*(k-1)+1):size_x*k);
 
            maskPostProcess(maskPostProcess~=1)=0;
            
            %Obtain the boundaries of the segmentation 
            bPostProcess=bwboundaries(maskPostProcess);
            boundary=bPostProcess{1};
            xPP=boundary(:,2);
            yPP=boundary(:,1);

            %Change the row and the columns of th mosaic if it's necessary
            row = fix((i-1)/rowMosaic);
            col = mod((i-1),rowMosaic);

            %plot
            hold on;
            plot(xPP + col*size_x,yPP + row*size_y,'.','MarkerSize',2,'Color','g')
            hold off;

        end
        
        %--Print the yellow lines to visualisize the images separately
        for j=1:(colMosaic-1)%number of images per column 
            hold on;
            plot(1:size_x*rowMosaic,size_y*j,'.','MarkerSize',2,'Color','y')
            hold off;
        end;
        for i=1:(rowMosaic-1)%number of images per row 
            hold on;
            plot(size_x*i,1:size_y*colMosaic,'.','MarkerSize',2,'Color','y')
            hold off;
        end; 
    
        ordre(1:imgsMOSAIC)=[];
    end