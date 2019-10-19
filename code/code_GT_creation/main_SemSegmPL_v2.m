function [ MaskSemantic ] = main_SemSegmPL_v2( I,name_file)
% THIS FUNCTION CREATES THE MASK PLAQUE IMAGES IN CA. THE LABELS:
% '0': Backgorund
% '1': Plaque
%
% INPUT:
% I:THE ORIGINAL IMAGE WITH THE MANUALLY DELINEATED LINES 
% name_img: The name of image that will be processed
%
% OUTPUT:
% MaskSemantic: THE MASK RESULT.    

    
    % figure();imshow(I); %%% FIGURE
    Ic=I;

    % FIND THE MANUAL SEGMENTATION
    % The pixels fom the manual line aren't in a grayscale range
    IR=Ic(:,:,1);
    IG=Ic(:,:,2);
    IB=Ic(:,:,3);
    
    green = max(max(IG-IR));
    if green < 100
    
%     Iseg1=IG-IR;
%     Iseg2=IG-IB;
%     Iseg=Iseg1+Iseg2;
        Iseg1=IB-IR;
        Iseg2=IB-IG;
        Iseg=Iseg1+Iseg2;
    % To properly work, letters should be dropped out
    
          %figure(); imshow(Iseg) %%% FIGURE
          %figure(); imshow(IB) %%% FIGURE



    % BINARIZE THE IMAGE   
%       T=0.05;%Treshold for binarizing
        T=0.02;%Treshold for binarizing
        Isegb=im2bw(Iseg,T); %binarizing the image with a theshold T
    
       % figure(); imshow(Isegb)%%% FIGURE
    else 
        
        Iseg1=IG-IR;
        Iseg2=IG-IB;
        Iseg=Iseg1+Iseg2;
       % figure(); imshow(Iseg) %%% FIGURE
        T=0.05;
        Isegb=im2bw(Iseg,T);
    end
    %MORPHOLOGICAL CLOSING 
%     cl1=4;cl2=2;%Rectangle for morphological closing
    cl1=20;cl2=20;%Rectangle for morphological closing
    SE=strel('rectangle',[cl1 cl2]);
    Isegb=imclose(Isegb,SE);
   
    % figure(); imshow(Isegb)%%% FIGURE      
    
    % INVERSE THE IMAGE IN ORDER TO FIND THE COMPONENTS    
    INV=zeros(size(Isegb));    
    INV(find(Isegb==0))=1;
    Isegb=INV;

    %get the connected components from the binary inverted image

    CC = bwconncomp(Isegb);    
    % figure(); imshow(Isegb); %%% FIGURE
    
    
    %INVERSE AGAIN THE IMAGE IN ORDER TO FIND THE COMPONENTS
    INV=zeros(size(Isegb));
    INV(find(Isegb==0))=1;
    
    % figure(),  imshow(INV); %%% FIGURE
    
    %Fill the regions
    Ifill=imfill(INV);
    % figure(),  imshow(Ifill); %%% FIGURE
    
    
    %get the connected components from the Mask image
    CC = bwconncomp(Ifill);
    
    %The 2 biggest components: The Letters and the Plaque
    [y1,x1]=getComponent(Ifill,CC,1);%
%     [y2,x2]=getComponent(Ifill,CC,2);% assuming that the bigges is the plaque ??
    
%     %Choose the PLaque
%      if min(y1)<=min(y2)
%      %if min(y1)>min(y2)
%          Ifill(y1,x1)=0;
%      else
%          Ifill(y2,x2)=0;         
%      end;
%      
     % figure(),  imshow(Ifill); %%% FIGURE
 
    
    %SMOOTH THE CONTOUR OF THE PLAQUE
    
    cl1=5;cl2=5;%Rectangle for morphological closing
    SE=strel('rectangle',[cl1 cl2]);
    maskPL=imclose(Ifill,SE);    
   %  figure(); imshow(maskPL)  %%% FIGURE

    cl1=5;cl2=5;%Rectangle for morphological open
    SE=strel('rectangle',[cl1 cl2]);
    maskPL=imopen(maskPL,SE);   
    % figure(); imshow(maskPL)  %%% FIGURE


    %%% IN CASE THERE ARE MORE THAN 1 COMPONENT,CHOOSE THE BIGGEST ONE
    CC = bwconncomp(maskPL);%Get components    
    [yPL,xPL]=getComponent(maskPL,CC,1);%
    maskPL=zeros(size(maskPL));
    IND=sub2ind(size(maskPL),yPL,xPL);
    maskPL(IND)=1;
        
    MaskSemantic=maskPL;
    cl1=7;cl2=7;
    SE=strel('rectangle',[cl1 cl2]);
    MaskSemantic=imerode(maskPL,SE);

    % figure(); imshow(MaskSemantic)  %%% FIGURE

% afegir erosio per arrodonir 

   