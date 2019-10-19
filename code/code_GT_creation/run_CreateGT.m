% THIS SCRIPT GENERATES THE GT FOR THE SEMANTIC SEGMENTATION IN BULB
% IMAGES. FROM THE MANUALLY SEGMENTED IMAGES IT CREATES A MATRIX IMAGE ONLY
% WITH LABELS FROM THE REGIONS
%
% THIS SEMANTIC IMAGE HAS 2 LABELS:
% 1-ROI: PLaque or IMT REgion
% 2-Background
%
% OPTIONALLY:
% -YOU MAY SHOW THE MOSAICS WITH BOTH IMAGES: ORIGINAL AND LABELED
% -YOU MAY STORE THE PNG FILE WITH THE MASK IMAGE
%
clear;
close all;


%%%%%%%%%%%%%
%%% ADD PATHS

addpath('functions_show\');%Functions for showing the moasic results


%%%%%%%%%%%%%%%%%%%%%%%%
%%% INITIALIZE VARIABLES

% CHOOSE THE DICOM OR PNG IMAGES
% '1' Dicom
% '2' PNG image
TYPE=2;

% CHOOSE THE PLAQUE OR IMT REGION IS SEGMENTED
% '1' Plaque image
% '0' IMT region
PLAQUE=1;

%SET THE DIRECTORI WHERE THE MANUALLY GT IS STORED
    if PLAQUE==1
       %path_dir_ORIGIN='Manual_GT_PL_\im_JON\PL\';
       %path_dir_ORIGIN='Manual_GT_PL_\SEPARADES_FINALS_ID\parelles_pl\PL\';
       %path_dir_ORIGIN = 'Manual_GT_PL_\NEFRONA_crops\ACC\parelles_pl\PL\'
       path_dir_ORIGIN = 'Manual_GT_PL_\GT_FEM_PL_20180620_no_yellow\'
       %path_dir_DEST_GT='GT_PL/';
       %path_dir_DEST_GT='GT_PL_FINAL_20180607/';
       path_dir_DEST_GT='GT_PL_FEM_20180620\';
       path_dir_DEST_GT_v2='GT_PL_FEM_20180620_v2\';
    elseif PLAQUE==0
       %path_dir_ORIGIN='Manual_GT_PL_\im_JON\IMT\';
       %path_dir_ORIGIN='Manual_GT_PL_\im_JON\test\';
       path_dir_ORIGIN='Manual_GT_PL_\\SEPARADES_FINALS_ID\parelles_itm\IMT\';
       %path_dir_DEST_GT='GT_IMT/';
%        path_dir_DEST_GT='GT_IMT_FINAL_20180607/';
       path_dir_DEST_GT='GT_IMT_FINAL_20180618_v2\';
    end;


%path_dir_ORIGIN='ManualGT_PL\';
dir_files=dir(strcat(path_dir_ORIGIN));%Read the (Dicom) files


% SET THE DIRECTORY WHERE THE 6 LABEL MASK'S WILL BE STORED
%path_dir_DEST_GT='GT_/';



STORE=1; % '1' means the result will be stored, '0' otherwise
MOSAIC=0; % '1' means the result will be shown in a mosaic, '0' otherwise

% INITIALIZE VARIABLE FOR THE MOSAICS
if MOSAIC==1
    stripIss=[];%Initialize the strips for the mask result
    stripOri=[];%Initialize the strips for the original image
    len=10;%The number of images to show in one mosaic
   
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CREATE THE LABEL MASK FOR EACH IMAGE

INI=3; %3 as minimum, since we didn't spcefiy the file extension
FIN=length(dir_files);

%Variables to count the number of images and mosaics
cont=0;
numMosaic=0;

for  k=INI:FIN


    %%%%%%%%%%%%%%%%%%
    %%% READ THE IMAGE 

    name_file=dir_files(k).name;%obtain the name of the subfolder
    if name_file(1:2)=='._' 
        name_file = extractAfter(dir_files(k).name,'._')
    end
    %name_file = '105202_E_0.png'
    pth_file=strcat(path_dir_ORIGIN,name_file);
    %I = dicomread(pth_file); 
    disp(pth_file)
    I = imread(pth_file);
%     figure(); imshow(I); %%% FIGURE 

    %CROP THE IMAGE: CHANGE THAT VALUES!!!
    xmin=144 ;
    ymin=22;
    width= 350;
    height= 350;
    %I=imcrop(I,[xmin ymin width height]);
    %I=imcrop(I);

%     figure(); imshow(I); %%% FIGURE 

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% GET THE MASK SEGMENTATION

    if PLAQUE==1
        [ MaskSemantic ] = main_SemSegmPL_v2( I ,name_file); % GET THE 6 LABEL'S MASK FROM THE IMAGE
    elseif PLAQUE==0
        [ MaskSemantic ] = main_SemSegmIMT_v2( I ,name_file); % GET THE 6 LABEL'S MASK FROM THE IMAGE
    end;
      
%     figure(); imshow(MaskSemantic); %%% FIGURE 
      
    Issnew= getSegmentationColors( I,MaskSemantic );% GET THE IMAGE WITH A DIFFERENT COLOR FOR EACH REGION
  
%       figure();imshow(Issnew); %%% FIGURE
    
    if MOSAIC==1    
        stripIss=[stripIss,Issnew];% Save the mask segmetnation in the strip image
        stripOri=[stripOri,I];% Save the original image in the strip image
        %         figure(); imshow(stripIss); 
        
       %Actualize the counting number for the imafges in a mosaic
        cont=cont+1;
    
        %Show the mosaic results for the first nimages
        if cont==len
            
            numMosaic=numMosaic+1;
            
            direction=1;
            nimages=len; %Indicates one mosaic with 'len' number of images
            showResultsSS(stripOri,stripIss,size(I,2),size(I,1),nimages,len,direction);
            
            %For only ine image:
%             title(strcat('K=',int2str(k),' ImgName: ',name_file));
         
            %Save the current mosaic in a png file
            name_fig='Nefrona_GT_mos';
            saveName=strcat(name_fig,int2str(numMosaic),'.png');
            saveas(gcf,saveName);
            close all;

            cont=0; %reset the variable for counting tihe images in a mosaic
            stripIss=[];
            stripOri=[];
        end;
    end;
    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%% STORE THE MASK SEGMENTATION
    
    if(STORE==1)
          
        %GT Image
        path_DESTINATION=strcat(path_dir_DEST_GT_v2,name_file);
        path_DESTINATION=strcat(path_DESTINATION,'_v2','.png');
        b = MaskSemantic;
        imwrite(b,path_DESTINATION);   
        grayImage = (255* uint8(b));
        RGB = cat(3, grayImage, grayImage, grayImage);
        
        % original mar
        path_DESTINATION=strcat(path_dir_DEST_GT,name_file);
        path_DESTINATION=strcat(path_DESTINATION);% Add the '.png' extension
        b = cast(MaskSemantic,'uint8');
        class(b);
        imwrite(b,path_DESTINATION);  
        
    end

end;





