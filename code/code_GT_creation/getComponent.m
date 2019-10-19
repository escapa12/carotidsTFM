function [y,x ] = getComponent( Im,CC,idx )
% CC: List of the connected components
% idx: The number of component

    numPixels = cellfun(@numel,CC.PixelIdxList);
    [s,indxs] = sort(numPixels,'descend'); 
    idxLargest=CC.PixelIdxList{indxs(idx)};%indexes corresponding to the component
    S=size(Im);
    [y,x]=ind2sub(S,idxLargest); %convert index to coordinates



end

