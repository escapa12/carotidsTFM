function [ MaskSemantic ] = main_SemSegmIMT( I,name_file )
% THIS FUNCTION CREATES THE MASK OF THE IMT REGION IN CA THE LABELS:
% '0': Backgorund
% '1': IMT region
%
% INPUT:
% I:THE ORIGINAL IMAGE WITH THE MANUALLY DELINEATED LINES 
% name_img: The name of image that will be processed
%
% OUTPUT:
% MaskSemantic: THE MASK RESULT.

%     figure();imshow(I); %%% FIGURE
    Ic=I;

    % FIND THE MANUAL SEGMENTATION
    % The pixels fom the manual line aren't in a grayscale range
    IR=Ic(:,:,1);
    IG=Ic(:,:,2);
    IB=Ic(:,:,3);
    Iseg1=IG-IR;
    Iseg2=IG-IB;
    Iseg=Iseg1+Iseg2;

    %      figure(); imshow(Iseg) %%% FIGURE

    % BINARIZE THE IMAGE   
    T=0.2;%Treshold for binarizing
    Isegb=im2bw(Iseg,T); %binarizing the image with a theshold T
    
%     figure(); imshow(Isegb)%%% FIGURE
    

    %MORPHOLOGICAL CLOSING 
    cl1=1;cl2=3;%Rectangle for morphological closing     
    SE=strel('rectangle',[cl1 cl2]);
    Isegb=imclose(Isegb,SE);
   
%     figure(); imshow(Isegb)%%% FIGURE      
       
    %get the connected components from the binary image
    CC = bwconncomp(Isegb);    
    
    %REMOVE THE BIGGEST COMPONENT (IT CORRESPONDS TO THE LETTERS)
    [yL,xL]=getComponent(Isegb,CC,1);%
    mask=zeros(size(Isegb));
    IND=sub2ind(size(mask),yL,xL);
    mask(IND)=1;%Create the mask with the letter in order to remove them
    Isegb=Isegb-mask;%remove the letters
    
%     figure(); imshow(Isegb); %%% FIGURE
    
    
    %FIND LUMEN-INIMTA AND MEDIA-ADVENTITA INTERFACES BY LOOKING THE 
    % THE WIDH OF EACH COMPONBENT
    CC = bwconncomp(Isegb); %get the new connected components    
    dist1=0;
    indx1=0;
    dist2=0;
    indx2=0;
    for i=1:CC.NumObjects
        [y,x]=getComponent(Isegb,CC,i);%
        dist=max(x)-min(x);%Width of the component
        if dist>=dist1
            distaux=dist1;
            indxaux=indx1;
            dist1=dist;
            indx1=i;
            if distaux>=dist2
                dist2=distaux;
                indx2=indxaux;
            end
        elseif dist>=dist2
            dist2=dist;
            indx2=i;
        end;
    end;
    
    %Define the Mask
    mask=zeros(size(mask));

%   figure(); imshow(mask)  %%% FIGURE

 
    % CREATE THE MASK WITH ONLY LUMEN-INRIMA AND MEDIA-ADVENTITIA
    [y1,x1]=getComponent(Isegb,CC,indx1);%
    IND1=sub2ind(size(mask),y1,x1);
    mask(IND1)=1;
 
    [y2,x2]=getComponent(Isegb,CC,indx2);%
    IND2=sub2ind(size(mask),y2,x2);
    mask(IND2)=1;
    
        %     figure(); imshow(mask)  %%% FIGURE
  
    % FIND THE LEFT AND RIGHT LIMIT FOR THE IMT REGION. REMOVE THE POINTS OUTSIDE
    minx=min(min(x1),min(x2));
    mask(:,1:minx)=0;   
    maxx=min(max(x1),max(x2));
    mask(:,maxx:size(mask,2))=0;
   
    %     figure(); imshow(mask)  %%% FIGURE
    
    % CLOSE THE IMT REGION CREATING THE COUMNS IN THE LEFT AND RIGHT LIMITS
    limLeft=find(mask(:,(minx+1))==1)
    mask(limLeft(1):limLeft(2),(minx+1))=1;    
    limRight=find(mask(:,(maxx-1))==1)
    mask(limRight(1):limRight(2),(maxx-1))=1;
    
    %     figure(); imshow(mask)  %%% FIGURE
        
    %Fill the imt region
    mask=imfill(mask);
    
    MaskSemantic=mask;

%     figure(); imshow(MaskSemantic)  %%% FIGURE



end

