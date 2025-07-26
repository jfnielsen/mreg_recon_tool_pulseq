%%This is a demo MREG sequence
%
% Changes for GE 
%  - Add TRID label
%  - Don't include blocks containing only labels
%  - Set sys struct (raster times etc) for GE hardware
%  - Interpolate gradient waveform

vendor = 'GE';   % 'GE' or 'Siemens'

isGE = strcmp(upper(vendor(1)), 'G');
isSiemens = strcmp(upper(vendor(1)), 'S');

seq = mr.Sequence(); %create a new sequence object
FOV = 192;           %define FOV and resolution in mm
Resolution = 3;
TR = 100e-3;         %set TR in ms

%% set system limits
% the spoiler at the end often causes PNS problems, therefore the max slew
% rate can be set independently
%dG=250e-6;
maxGrad = 38;  % mT/m
rfDeadTime = 100e-6;
rfRingdownTime = 60e-6;
adcDeadTime = 40e-6;
adcRasterTime = 2e-6;
rfRasterTime = 2e-6;
gradRasterTime = 4e-6;
blockDurationRaster = 4e-6;
B0 = 3;    % T

%sys = mr.opts('MaxGrad', 38, 'GradUnit', 'mT/m', ...
%    'MaxSlew', 170, 'SlewUnit', 'T/m/s', 'rfRingdownTime', 10e-6, ...
%    'rfDeadTime', 100e-6, 'adcDeadTime', 10e-6);
sys = mr.opts('maxGrad', maxGrad, 'gradUnit','mT/m', ... 
              'maxSlew', 170, 'slewUnit', 'T/m/s', ... 
              'rfDeadTime', rfDeadTime, ...
              'rfRingdownTime', rfRingdownTime, ...
              'adcDeadTime', adcDeadTime, ...
              'adcRasterTime', adcRasterTime, ...
              'rfRasterTime', rfRasterTime, ...
              'gradRasterTime', gradRasterTime, ...
              'blockDurationRaster', blockDurationRaster, ...
              'B0', B0);

%sys_true = mr.opts('MaxGrad', 38, 'GradUnit', 'mT/m', ...
%    'MaxSlew', 200, 'SlewUnit', 'T/m/s', 'rfRingdownTime', 10e-6, ...
%    'rfDeadTime', 100e-6, 'adcDeadTime', 10e-6);
sys_true = mr.opts('maxGrad', maxGrad, 'gradUnit','mT/m', ... 
              'maxSlew', 200, 'slewUnit', 'T/m/s', ... 
              'rfDeadTime', rfDeadTime, ...
              'rfRingdownTime', rfRingdownTime, ...
              'adcDeadTime', adcDeadTime, ...
              'adcRasterTime', adcRasterTime, ...
              'rfRasterTime', rfRasterTime, ...
              'gradRasterTime', gradRasterTime, ...
              'blockDurationRaster', blockDurationRaster, ...
              'B0', B0);

%sys_spoiler = mr.opts('MaxGrad', 38, 'GradUnit', 'mT/m', ...
%    'MaxSlew', 60, 'SlewUnit', 'T/m/s', 'rfRingdownTime', 10e-6, ...
%    'rfDeadTime', 100e-6, 'adcDeadTime', 10e-6);
sys_spoiler = mr.opts('maxGrad', maxGrad, 'gradUnit','mT/m', ... 
              'maxSlew', 60, 'slewUnit', 'T/m/s', ... 
              'rfDeadTime', rfDeadTime, ...
              'rfRingdownTime', rfRingdownTime, ...
              'adcDeadTime', adcDeadTime, ...
              'adcRasterTime', adcRasterTime, ...
              'rfRasterTime', rfRasterTime, ...
              'gradRasterTime', gradRasterTime, ...
              'blockDurationRaster', blockDurationRaster, ...
              'B0', B0);

warning('OFF', 'mr:restoreShape')
%% RF 
% due to the imperfect slice profile of the excitation pulse the slice
% thickness for the pulse is set to 0.8*slicethickness for a more
% homogeneous excitation
flipangle = 25/180*pi;   %set flip angle in rad
sliceThickness = 150;    %set slice thickness (slab excitation)
[rf, gz, gzReph] = mr.makeSincPulse(flipangle,'system',sys,'use','excitation','Duration',1.4e-3,...
    'SliceThickness',0.001*sliceThickness*0.8,'apodization',0.5,'timeBwProduct',16);

%% create mreg trajectory & adc
%undersampling parameters can be varied (see stack_of_spirals.m), old
%settings are R = [3 6 2 5]. PSFs with the undersampling parameters can be
%simulated using the script stack_of_spirals_psf.m
R = [2.8687 6.0202 2.552 3.0584];

T = stack_of_spirals(R,1,1,0.001*FOV,0.001*Resolution,1); 
%TODO: interpolate to 4us
return
trajectStruct_export(T,'2025_Pulsec_SoS',1);% g unit T/m to mT/m, normalized
Grads_calc = -T.G'*sys.gamma;

adcSamples = length(Grads_calc)*2; % oversampling is usually 2
dW=sys.gradRasterTime/2;
% the adc should be splittable into N equal parts, each of which is aligned
% to the gradient raster. each segment however needs to have the number of
% samples divisible by 4 to be executable on siemens scanners
adcSamples=floor(adcSamples/8)*8; 
while adcSamples>0
    adcSegmentFactors=factor(round(adcSamples/2));
    assert(adcSegmentFactors(1)==2); 
    assert(adcSegmentFactors(2)==2); 
    assert(length(adcSegmentFactors)>3); 
    adcSegments=1;
    for i=3:length(adcSegmentFactors) 
        adcSegments=adcSegments*adcSegmentFactors(i);
        adcSamplesPerSegment=adcSamples/adcSegments;
        if (adcSamplesPerSegment<=8192)
            break
        end
    end
    if (adcSamplesPerSegment<=8192)
        break
    end
    adcSamples=adcSamples-8; %
end
assert(adcSamples>0); % we could not find a suitable segmentation...

adc = mr.makeAdc(adcSamples,'Duration',dW*adcSamples,'system',sys_true);

mregx = mr.makeArbitraryGrad('x',Grads_calc(1,:),'first',0, 'last',0,'Delay',adc.delay,'system',sys_true); % looks like the optimized MREG trajectory often exceeds the slew rate so we use an additional system with true maximum limits
mregy = mr.makeArbitraryGrad('y',Grads_calc(2,:),'first',0, 'last',0,'Delay',adc.delay,'system',sys_true);
mregz = mr.makeArbitraryGrad('z',Grads_calc(3,:),'first',0, 'last',0,'Delay',adc.delay,'system',sys_true);

%% Create fat-sat pulse 
% % not used
% % (in Siemens interpreter from January 2019 duration is limited to 8.192 ms, and although product EPI uses 10.24 ms, 8 ms seems to be sufficient)
B0=2.89; % 1.5 2.89 3.0
sat_ppm=-3.45;
sat_freq=sat_ppm*1e-6*B0*sys.gamma;
rf_fs = mr.makeGaussPulse(110*pi/180,'system',sys,'Duration',8e-3,...
    'bandwidth',abs(sat_freq),'freqOffset',sat_freq);
gz_fs = mr.makeTrapezoid('z',sys,'delay',mr.calcDuration(rf_fs),'Area',1/1e-4); % spoil up to 0.1mm

%% spoiler
gx_spoil=mr.makeTrapezoid('x',sys,'Area',-10*1000,'system',sys_spoiler);% spoilers cause a lot of PNS,therefore extra parameterset sys_spoiler
gy_spoil=mr.makeTrapezoid('y',sys,'Area',-10*1000,'system',sys_spoiler);
gz_spoil=mr.makeTrapezoid('z',sys,'Area',-10*1000,'system',sys_spoiler);


%trig = mr.makeDigitalOutputPulse('ext1','duration',100e-6); %'osc0', 'osc1', 'ext1' 
%% seq Block
seq = mr.Sequence(sys);
%seq.addBlock(rf_fs,gz_fs)
%seq.addBlock(gz_spoil)
seq.addBlock(rf,gz, mr.makeLabel('SET', 'TRID', 1));
seq.addBlock(gzReph)

%turn off FOV shifting for readout
nopos_label = mr.makeLabel('SET', 'NOPOS', 1);
if ~isGE
    seq.addBlock(nopos_label);
end
seq.addBlock(mregx,mregy,mregz,adc)
%switch back on for spoiler
nopos_label_off = mr.makeLabel('SET', 'NOPOS',0);
if ~isGE
    seq.addBlock(nopos_label_off);
end
seq.addBlock(gx_spoil,gy_spoil,gz_spoil)
delayTR = TR - mr.calcDuration(mregx,adc)-mr.calcDuration(rf,gz)-mr.calcDuration(gzReph)-mr.calcDuration(gz_spoil);
seq.addBlock(mr.makeDelay(delayTR));

TEs = mr.calcDuration(rf, gz) + mr.calcDuration(gzReph);
TEs(:,2) = T.TE  .* 0.000001 + TEs(:,1);
seq.setDefinition('TE',TEs);% TEs is the array of TE (max. array size = 32, only first 2 matter for MREG). Unit: second
seq.plot('showblocks',1)
%% new single-function call for trajectory calculation
%[ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing] = seq.calculateKspacePP();

%plot trajectory if wanted
% figure; hold on;
% plot3(ktraj(1,:),ktraj(2,:),ktraj(3,:),'b-')
% %plot3(ktraj_adc(1,:),ktraj_adc(2,:),ktraj_adc(3,:),'r'); 
%     axis equal
%     view([0 2])

%% PNS 
%opportunity to simulate pns for scanner used
%[pns_ok, pns_n, pns_c, tpns]=seq.calcPNS('MP_GPA_K2309_2250V_951A_AS82.asc'); % prisma Freiburg

%% prepare to write
[ok, error_report]=seq.checkTiming;

seq.setDefinition('FOV', 0.001*[FOV FOV sliceThickness]);
seq.setDefinition('MaxAdcSegmentLength', adcSamplesPerSegment); 
seq.setDefinition('Name', 'mreg_FOV');
%seq.setDefinition('ReceiverGainHigh',1) ;
seq.write('myMREG_FOV.seq');   % Output sequence for scanner
%seq.install('Siemens')
%rep = seq.testReport;
%fprintf([rep{:}]);
