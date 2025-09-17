addpath(genpath('..'));

% actions
createSequenceFile = true;
reconstruct = false;

fn = 'myMREG_FOV';

if createSequenceFile
    % create .seq file
    system('git clone --branch v1.4.2 git@github.com:pulseq/pulseq.git');
    addpath pulseq/matlab
    MREG_github_FOV;

    % Convert .seq file to a PulCeq (Ceq) object
    %system('git clone --branch v2.4.1 git@github.com:HarmonizedMRI/PulCeq.git');
    %system('git clone --branch tv7_dev git@github.com:HarmonizedMRI/PulCeq.git');
    %addpath PulCeq/matlab
    addpath ~/github/HarmonizedMRI/PulCeq/matlab

    ceq = seq2ceq([ fn '.seq']);

    % Check the ceq object:
    % Define hardware parameters, and
    % check if 'ceq' is compatible with the parameters in 'sys'
    psd_rf_wait = 58e-6;  % RF-gradient delay, scanner specific (s)
    psd_grd_wait = 60e-6; % ADC-gradient delay, scanner specific (s)
    b1_max = 0.25;         % Gauss
    g_max = 5;             % Gauss/cm
    slew_max = 20;         % Gauss/cm/ms
    gamma = 4.2576e3;      % Hz/Gauss
    sys = pge2.getsys(psd_rf_wait, psd_grd_wait, b1_max, g_max, slew_max, gamma);
    %pge2.validate(ceq, sys);

    pge2.plot(ceq, sys, 'timeRange', [0 inf]);

    % Write ceq object to file.
    % pislquant is the number of ADC events used to set receive gain in Auto Prescan
    writeceq(ceq, [fn '.pge'], 'pislquant', 1);
end

if reconstruct
    system('git clone --depth 1 --branch v1.9.0 git@github.com:toppeMRI/toppe.git');
    addpath toppe

    addpath ~/Programs/orchestra-sdk-2.1-1.matlab/

    archive = GERecon('Archive.Load', 'data.h5');

    % skip past receive gain calibration TRs (pislquant)
    for n = 1:pislquant
        currentControl = GERecon('Archive.Next', archive);
    end

    % read first phase-encode of first echo
    currentControl = GERecon('Archive.Next', archive);
    [nx1 nc] = size(currentControl.Data);
    ny1 = nx1;
    d1 = zeros(nx1, nc, ny1);
    d1(:,:,1) = currentControl.Data;

    % read first phase-encode of second echo
    currentControl = GERecon('Archive.Next', archive);
    [nx2 nc] = size(currentControl.Data);
    d2 = zeros(nx2, nc, ny1);

    for iy = 2:ny1
        currentControl = GERecon('Archive.Next', archive);
        d1(:,:,iy) = currentControl.Data;
        currentControl = GERecon('Archive.Next', archive);
        d2(:,:,iy) = currentControl.Data;
    end

    d1 = permute(d1, [1 3 2]);   % [nx1 nx1 nc]
    d2 = permute(d2(:, :, end/2-nx2:end/2+nx2-1), [1 3 2]);   % [nx2 nx2 nc]

    [~, im1] = toppe.utils.ift3(d1, 'type', '2d');
    [~, im2] = toppe.utils.ift3(d2, 'type', '2d');

    system('git clone --depth 1 git@github.com:JeffFessler/mirt.git');
    cd mirt; setup; cd ..;

    subplot(121); im(im1); title('echo 1 (192x192, dwell = 20us)');
    subplot(122); im(im2); title('echo 2 (48x192, dwell = 40us)');
end

