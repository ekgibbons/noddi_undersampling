function [TrnsfrmdData] = loadH5DataMod(prefixName,DataPath,FileName)
%[DATA] = loadH5Data(prefixName) 
%load h5 data from the prefix specified. May need to modify this code if
%the suffix of the .h5 data changes. Outputs a cell array containing all of
%the data

pathStart = pwd;
if nargin<2
    DataPath= '/v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/Stroke_patients/Stroke_DSI_Processing/Data/Models/EricProcessing';
end
if nargin<3
    FileName='24_directions_2d.h5';
end

%May need to change the file name 
% $$$ h5disp(sprintf('%s/%s_%s',DataPath,prefixName,FileName)); % helps to see the DATASET name predictions
DATA = h5read(sprintf('%s/%s_%s',DataPath,prefixName,FileName),'/predictions');

for k = 1:4
    AllNewData{k} = DATA(k,:,:,:);
    AllNewData{k} = (squeeze(((AllNewData{k}))));
end

% to rearrange properly:
for trnsfrmIdx = 1:4
    G = AllNewData{trnsfrmIdx}; 
    temp = zeros(128,128,size(G,3));
    for k = 1:51
        temp(:,:,k) = ((G(k,:,:)));
    end
    TrnsfrmdData{trnsfrmIdx} = double(rot90(temp(128:-1:1,:,:),-1)); %aligns the python
    %way of storing with matlab so all the labels line up correctly
end

