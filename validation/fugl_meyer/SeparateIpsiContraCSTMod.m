function [ipsi contra] = SeparateIpsiContraCSTMod(prefix,ipsi_contra)
%[ipsi contra] = SeparateIpsiContraCST(prefix,ipsi_contra)
%prefix refers to the prefix identifying the motor labels, should be a cell
%array. ipsi_contra if the left side is the ipsilesional, put 1, if right
%side is ipsilesional put -1. P032315 should be -1. 

original_path = pwd;

cd /v/raid1b/khodgson/MRIdata/DTI/CNC_Imris/Stroke_patients/Stroke_DSI_Processing/Data/Models


%labels = {'07', '08', '15', '16', '19', '20' '25' '26'};
%AUG 28 Exclude 07 and 08 because data bad
labels ={'07','08', '15', '16', '19', '20' '25' '26'};
pref= 'FA_label';
suff = '.nii.gz';

if nargin<1
    prefix = {'P041714','P081114','P010715','P061114','P030315','P032315','P080715',...
              'P020916','P111816'};
end

if nargin<2
    ipsi_contra = [1 -1 -1 1 -1 -1 1 -1 -1]; % indicates which side is ipsi
end

ipsi = cell(length(labels)/2,1);
contra = cell(length(labels)/2,1);
for k = 1:length(prefix)        
    if exist(['MotorTractLabels/',prefix{k},'_',pref,labels{1},suff])
        for ii=1:length(labels)
            if ii==1
                clear LABEL
            end
            %cd to correct directory
            %redo all the motor tract analysis? probably yes
            L = load_nifti(['MotorTractLabels/',prefix{k},'_',pref,labels{ii},suff]); % may need to change names
                                                                                      %         fromslice = 2; %starts on the slice after the number listed here
                                                                                      %         numpixelsperslice = 128*128;
                                                                                      %         L.vol(1:fromslice*numpixelsperslice) = 0; %AUG 28 should include only data after slice 6. 128*128*6 = 98304, or 128*128*4 = 65536%clear L
                                                                                      %         endslice = 39*128*128;
                                                                                      %         L.vol(endslice:end) = 0;

            LABEL{ii} = find(L.vol);
            %LABEL{ii} = L.vol>0;
            

        end
    end
    % separate labels into ipsi and contra segments
    odd = 1;
    even = 2;
    for hemisphere_index = 1:length(labels)/2
        if ipsi_contra(k)>0
            ipsi{hemisphere_index} = LABEL{even};
            contra{hemisphere_index} = LABEL{odd};
            odd = odd+2;
            even = even+2;
        else
            ipsi{hemisphere_index} = LABEL{odd};
            contra{hemisphere_index} = LABEL{even};
            odd = odd+2;
            even = even+2;
        end
    end

end

cd_command = sprintf('cd %s', original_path)

system(cd_command)
cd(original_path)
pwd