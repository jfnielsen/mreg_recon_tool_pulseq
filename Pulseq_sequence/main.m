if ~exist('mr.Sequence');
    setup;
end

fn = 'mreg';

% write the .seq file
MREG_github_FOV;

% convert to ceq object
ceq = seq2ceq([fn '.seq']); %, 'usesRotationEvents', true);

% Check the ceq object:
% First define hardware parameters
psd_rf_wait = 100e-6;  % RF-gradient delay, scanner specific (s)
psd_grd_wait = 100e-6; % ADC-gradient delay, scanner specific (s)
b1_max = 0.25;         % Gauss
g_max = 5;             % Gauss/cm
slew_max = 20;         % Gauss/cm/ms
coil = 'xrm';          % MR750. See pge2.opts()
sysGE = pge2.opts(psd_rf_wait, psd_grd_wait, b1_max, g_max, slew_max, coil);

% check if 'ceq' is compatible with the parameters in 'sys'
% checks PNS and b1/gradient limits,
pars = pge2.check(ceq, sysGE);

% Plot the beginning of the sequence
%S = pge2.plot(ceq, sysGE, 'timeRange', [0 0.1], 'rotate', false);

% Check mechanical resonances (forbidden frequency bands)
S = pge2.plot(ceq, sysGE, 'timeRange', [0 0.1], 'rotate', true, 'interpolate', true);
%check_grad_acoustics(S.gx.signal/100, 'xrm');   % MR750: 'xrm'; UHP: 'hrmbuhp'

% Write ceq object to file.
% pislquant is the number of ADC events used to set Rx gains in Auto Prescan
writeceq(ceq, [ fn '.pge'], 'pislquant', 10);

return

% After simulating in WTools/VM or scanning, grab the xml files 
% and compare with the seq object, e.g.:
seq = mr.Sequence();
seq.read([fn '.seq']);
warning('OFF', 'mr:restoreShape');  % turn off Pulseq warning for spirals
xmlPath = '~/transfer/xml/';
%pge2.validate(ceq, sysGE, seq, xmlPath, 'row', [], 'plot', true);

