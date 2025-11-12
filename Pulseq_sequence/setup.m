addpath(genpath('..'));

addpath ~/github/jfnielsen/scanLog/Pulseq/forbiddenFreqs/  % check_grad_acoustics.m

system('git clone --branch v1.4.2 git@github.com:pulseq/pulseq.git');
addpath pulseq/matlab

%system('git clone --branch v2.4.1 git@github.com:HarmonizedMRI/PulCeq.git');
%system('git clone --branch tv7_dev git@github.com:HarmonizedMRI/PulCeq.git');
%addpath PulCeq/matlab
addpath ~/github/HarmonizedMRI/PulCeq/matlab

