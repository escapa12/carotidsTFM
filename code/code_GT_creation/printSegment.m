function [ Iss ] = printSegment( Iss,xl,yl,R,G,B )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here


for i=min(xl):max(xl)
    yl_col=yl(find(xl==i));
    Iss(min(yl_col):max(yl_col),i,1)=R;
    Iss(min(yl_col):max(yl_col),i,2)=G;
    Iss(min(yl_col):max(yl_col),i,3)=B;
   
end;

end

