function [UEtp2Prediction, PredictFromOriginal, UEtp2Actual] = ...
        predict_fm_score(patient_number, model_type, n_directions)
% Check whether the asymmetries in the Posterior Limb of the
% Internal Capsule (PLIC) still predict outcomes based on a baseline scan. 
% Using the linear regression equation from prior work, we input the
% difference of the mean in the ipsilesional PLIC and the contralesional
% PLIC defined as mean(contra) - mean(ipsi) to predict upper extremity
% outcomes.

% If you want to build a loop to check multiple cases and multiple
% undersampling scenarios it would be pretty easy. Loop through the
% prefixOptions and the FileName. 

%To calculate a the predicted score for a particular data set, change the
%StrokeNumber to the desired prefixOptions

original_path = pwd;

prefixOptions = {'P041714','P081114','P010715','P061114','P030315','P032315','P080715',...
                 'P020916','P111816'};

StrokeNumber = find(contains(prefixOptions,patient_number));

prefix = {prefixOptions{StrokeNumber}};

%load the undersampled data
DataPath = '/v/raid1b/egibbons/MRIdata/DTI/noddi/processing';
FileName = sprintf('%i_directions_%s.h5',n_directions,model_type);
[ImagingParameterMaps] = loadH5Data(prefix{1},DataPath,FileName); 

%separate the ipsilesional and contralesional CST hemispheres
ipsi_contra = [1 -1 -1 1 -1 -1 1 -1 -1]; % indicates which side is ipsi
[IpsiCSTcoord, ContraCSTcoord] = SeparateIpsiContraCST(prefix,ipsi_contra(StrokeNumber));

%Load the Original Data
[MeanDifferenceUndersampled] = MeanDifferenceCST(IpsiCSTcoord,...
                                                 ContraCSTcoord,...
                                                 ImagingParameterMaps);
ODI = load_nifti(sprintf('NODDIMaps/%s_hydi_odi.nii',prefix{1}));
RDI = load_nifti(sprintf('NODDIMaps/%s_hydi_ficvf.nii',prefix{1}));
CSF = load_nifti(sprintf('NODDIMaps/%s_hydi_fiso.nii',prefix{1}));
load(sprintf('GFAMaps/GFA_eddy_rot_bvecs/%s_GFA.mat',prefix{1}));
OriginalMaps = {ODI.vol, RDI.vol, CSF.vol,GFA};
[MeanDifferenceOriginal] = MeanDifferenceCST(IpsiCSTcoord,...
                                             ContraCSTcoord,...
                                             OriginalMaps);

%linear regression model parameters
ODImeanSlope = 423.18;
ODImeanIntercept = 59.23;
%predict FM UE tp2 from undersampled
UEtp2Prediction = ODImeanSlope*MeanDifferenceUndersampled(9) + ODImeanIntercept;
%(9) refers to ODI of internal capsule

% load the actual FM scores
load UEtp2.mat
PredictFromOriginal = ODImeanSlope*MeanDifferenceOriginal(9) + ODImeanIntercept;
UEtp2Actual = UEtp2(StrokeNumber);


cd(original_path);

end