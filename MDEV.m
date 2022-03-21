% calculate the maps of magnitude and the phase of the complex shear modulus and the amplitude of
% the shear wave field from the MRE-phase. This is done by smoothing
% unwrapping, filtering and MDEV-inversion.
%
% [absG, phi, strain] = MDEV(temporalWaveField, frequency, inplaneResolution, parameters)
%
% input:
% phase                      - 6D MRE-phase signal scaled over 2*pi
%                               - 1st and 2nd dimensions correspond to the in-plane matrix
%                                 size(number of rows and columns)
%                               - 3rd dimension corresponds to the number of slices
%                                 (at leeat 1 slice)
%                               - 4th dimension corresponds to the the number of timesteps
%                                 (at least 3 timesteps)
%                               - 5th dimension contains the components of the displacement
%                                 the order of the components SS, RO, and PE is arbitrary
%                               - 6th direction corresponds to the number
%                               of frequencies
%
% inplaneResolution             pixel spaceing in meters, correspond to the 1st and
%                               2nd dimensions in MRsignal, respectively.
%
% frequency                     array of frequencies in Hz
%                               note: length(frequency) must be identical to size(MRsignal,6)
%                               note: order of frequency must match the data order in the 6th dimension of MRsignal
%
%                               phase: 6D-array of wrapped MRE phase images in the following order
%
% parameters                    optional input argument for postprocessing parameters
% 
% output:
% absG                          map of the magnitude of complex shear modulus
%                               in Pa
%
% phi                           map of the phase of complex shear modulus
%                               in rad
% strain                        map of the in-plane strain
%                               in rad/m
%
% details to the inversion can be found in
% Streitberger et al.
% High-Resolution Mechanical Imaging of Glioblastoma by Multifrequency Magnetic Resonance Elastography.
% PLoS One. 2014 Oct 22;9(10):e110588
% http://dx.doi.org/10.1371/journal.pone.0110588
%
% details to the strain calculation can be found in
% Papazoglou et al.
% Scatter-based magnetic resonance elastography.
% Phys Med Biol. 2009 Apr 7;54(7):2229-41

% edit by Ingolf Sack, Carité-Universitätsmedizin Berlin, Berlin, Germany
% edit: 2013/11/26, Ingolf Sack
% last change: 2019/04/26, Heiko Tzschätzsch

function MDEV()

	load('data/BIOQIC/phantom_unwrapped_dejittered.mat');
	phase = phase_unwrap_noipd;
	frequency = info.frequencies_Hz;
	inplaneResolution = [info.dy_m, info.dx_m];

    % filter parameters
    if ~exist('parameters', 'var')
        parameters.smoothPhase.filter = 'gaussian';
        parameters.smoothPhase.size = [5 5 1]; %[pixel]
        parameters.smoothPhase.sd = 0.65; %[pixel] standard deviation
        parameters.fft.numberOfHarmonics = 1; %number of the harmonic after temporal FT
        parameters.radialFilter.lowpassThreshold = 100; %[1/m] butterworth threshold
        parameters.radialFilter.lowpassOrder = 1; %butterworth order
        parameters.laplaceInversion.density = 1000; %[kg/m^3] tissue density
        parameters.laplaceInversion.noiseCorrection = false; % enable noise correction
    end

    % smooth the phase signal
    smoothedPhase = smoothPhase(phase, (parameters.smoothPhase));

    % unwrap the phase and perform temporal Fourier transformation
    waveField = gradwrapFFT(smoothedPhase, inplaneResolution, (parameters.fft));

    % lowpass filtering using butterworth
    shearWaveField = radialFilter(waveField, inplaneResolution, (parameters.radialFilter));

    % main MDEV inversion using Laplace operator
    [absG, phi, strain] = laplaceInversion(shearWaveField, inplaneResolution, frequency, (parameters.laplaceInversion));

    phase_smoothed = smoothedPhase;
    save('data/BIOQIC/phantom_smoothed.mat', 'magnitude', 'phase_smoothed');

    magnitude = resample(magnitude, 1, 4, 'Dimension', 4);
    wave = waveField;
    wave_shear = shearWaveField;
    'shearWaveField real'
    [min(real(shearWaveField(:))), max(real(shearWaveField(:)))]
    'shearWaveField imag'
    [min(imag(shearWaveField(:))), max(imag(shearWaveField(:)))]
    save('data/BIOQIC/phantom_wave.mat', 'magnitude', 'wave');
    save('data/BIOQIC/phantom_wave_shear.mat', 'magnitude', 'wave_shear');

    magnitude = mean(magnitude, [4, 5, 6]);
    save('data/BIOQIC/phantom_MDEV.mat', 'magnitude', 'absG', 'phi', 'strain');
end

function smoothedPhase = smoothPhase(phase, parameters)

    % get the dimensions
    nSlice = size(phase,3);% number of slices
    nTimestep = size(phase,4);% number of time steps
    nComponent = size(phase,5);% number of components
    nFrequency = size(phase,6);% number of frequencies

    % loop for smoothing
    smoothedPhase = zeros(size(phase));
    for iFrequency = 1 : nFrequency
        for iComponent = 1: nComponent
            for iTimestep = 1 : nTimestep
                
                % create complex signal, smooth and extract the phase
                currentPhase = exp(1i*phase(:,:,:,iTimestep,iComponent,iFrequency));
                
                if nSlice > 1
                    temp = smooth3(currentPhase, parameters.filter, parameters.size, parameters.sd);
                else % nSlice == 1
                    currentPhase = repmat(currentPhase,[1 1 2]);
                    temp = smooth3(currentPhase, parameters.filter, parameters.size, parameters.sd);
                    temp = temp(:,:,1);
                end
                
                smoothedPhase(:,:,:,iTimestep,iComponent,iFrequency) = angle(temp);
                
            end
        end
    end
end

function waveField = gradwrapFFT(smoothedPhase, inplaneResolution, parameters)

    % get the dimensions
    n1 = size(smoothedPhase,1);% number of rows
    n2 = size(smoothedPhase,2);% number of columns
    nSlice = size(smoothedPhase,3);% number of slices
    nTimestep = size(smoothedPhase,4);% number of time steps
    nComponent = size(smoothedPhase,5);% number of components
    nFrequency = size(smoothedPhase,6);% number of frequencies

    % loop for unwrapping and fft
    gradient1 = zeros(n1, n2, nSlice, nTimestep);
    gradient2 = zeros(n1, n2, nSlice, nTimestep);
    waveField = zeros(n1, n2, nSlice, 2, nComponent, nFrequency);
    for iFrequency = 1 : nFrequency
        for iComponent = 1 : nComponent
            
            for iTimestep = 1 : nTimestep
                currentData = smoothedPhase(:,:,:,iTimestep,iComponent,iFrequency);
                
                if nSlice > 1
                    [temp2, temp1] = gradient(exp(1i*currentData), inplaneResolution(2), inplaneResolution(1), 1);
                else % nSlice == 1
                    [temp2, temp1] = gradient(exp(1i*currentData), inplaneResolution(2), inplaneResolution(1));
                end
                
                % in-plane derivative components
                temp = exp(-1i*currentData);
                gradient1(:,:,:,iTimestep) = imag(temp1 .* temp);
                gradient2(:,:,:,iTimestep) = imag(temp2 .* temp);
            end
            
            % fourier transformation and selection of harmonic
            fourier1 = fft(gradient1,[],4);
            waveField(:,:,:,1,iComponent,iFrequency) = fourier1(:,:,:,1+parameters.numberOfHarmonics);
            fourier2 = fft(gradient2,[],4);
            waveField(:,:,:,2,iComponent,iFrequency) = fourier2(:,:,:,1+parameters.numberOfHarmonics);
            
        end
    end
end

function shearWaveField = radialFilter(waveField, inplaneResolution, parameters)

    'waveField real'
    [min(real(waveField(:))), max(real(waveField(:)))]
    'waveField imag'
    [min(imag(waveField(:))), max(imag(waveField(:)))]

    % get the dimensions
    n1 = size(waveField,1);% number of rows
    n2 = size(waveField,2);% number of columns
    nSlice = size(waveField,3);% number of slices
    nGrad = size(waveField,4);% number of gradients
    nComponent = size(waveField,5);% number of components
    nFrequency = size(waveField,6);% number of frequencies

    % calculate the wavenumber for k-space filtering
    k1 = -( (0:n1-1)-fix(n1/2) ) / (n1*inplaneResolution(1));%[1/m] wavenumber in 1st direction
    k2 = ( (0:n2-1)-fix(n2/2) ) / (n2*inplaneResolution(2));%[1/m] wavenumber in 2nd direction
    absK = hypot( ones(n1,1)*k2, k1'*ones(1,n2) );%[1/m] transform into polar coordinates

    filter = 1 ./ (1 + (absK/parameters.lowpassThreshold).^(2*parameters.lowpassOrder));
    filter = ifftshift(filter);

    % loop for filtering
    shearWaveField = zeros(size(waveField));
    for iFrequency = 1 : nFrequency
        for iComponent = 1 : nComponent
            for iGrad = 1 : nGrad
                for iSlice = 1 : nSlice
                    
                    currentWaveField = waveField(:, :, iSlice, iGrad, iComponent, iFrequency);
                    
                    % filtering in k-space
                    data = fftn(currentWaveField);
                    filteredData = data .* filter;
                    currentShearWaveField = ifftn(filteredData);
                    
                    shearWaveField(:, :, iSlice, iGrad, iComponent, iFrequency) = currentShearWaveField;
                    
                end
            end
        end
    end

    'shearWaveField real'
    [min(real(shearWaveField(:))), max(real(shearWaveField(:)))]
    'shearWaveField imag'
    [min(imag(shearWaveField(:))), max(imag(shearWaveField(:)))]
end

function [absG, phi, strain] = laplaceInversion(shearWaveField, inplaneResolution, frequency, parameters)

    % get the dimensions
    n1 = size(shearWaveField,1);% number of rows
    n2 = size(shearWaveField,2);% number of columns
    nSlice = size(shearWaveField,3);% number of slices
    nGrad = size(shearWaveField,4);% number of gradients
    nComponent = size(shearWaveField,5);% number of components
    nFrequency = size(shearWaveField,6);% number of frequencies

    % loop for MDEV-inversion
    numeratorPhi = zeros(n1,n2,nSlice);
    denominatorPhi = zeros(n1,n2,nSlice);
    numeratorAbsG = zeros(n1,n2,nSlice);
    denominatorAbsG = zeros(n1,n2,nSlice);
    strain = zeros(n1,n2,nSlice);
    for iFrequency = 1 : nFrequency
        for iComponent = 1 : nComponent
            for iGrad = 1 : nGrad
                
                U = shearWaveField(:, :, :, iGrad, iComponent, iFrequency);
                
                if nSlice > 1
                    [gradient2, gradient1] = gradient(U,inplaneResolution(2), inplaneResolution(1), 1);
                    [~, gradient11] = gradient(gradient1,inplaneResolution(2), inplaneResolution(1), 1);
                    [gradient22, ~] = gradient(gradient2,inplaneResolution(2), inplaneResolution(1), 1);
                else % nSlice == 1
                    [gradient2, gradient1] = gradient(U,inplaneResolution(2), inplaneResolution(1));
                    [~, gradient11] = gradient(gradient1,inplaneResolution(2), inplaneResolution(1));
                    [gradient22, ~] = gradient(gradient2,inplaneResolution(2), inplaneResolution(1));
                end
                
                % laplace of U
                LaplaceU = gradient11 + gradient22;
                
                % calculation of numerator and denominator for MDEV-inversion
                numeratorPhi = numeratorPhi + real(LaplaceU).*real(U) + imag(LaplaceU).*imag(U);
                denominatorPhi = denominatorPhi + abs(LaplaceU).*abs(U);
                
                numeratorAbsG = numeratorAbsG + parameters.density*(2*pi*frequency(iFrequency)).^2.*abs(U);
                denominatorAbsG = denominatorAbsG + abs(LaplaceU);
                
                % sum the strain
                strain = strain + abs(U);
                
            end
        end
    end

    % avoid division by zero
    denominatorPhi(denominatorPhi == 0) = eps;
    denominatorAbsG(denominatorAbsG == 0) = eps;

    % inversion
    phi = acos(-numeratorPhi./denominatorPhi); %[rad]
    if isfield(parameters,'noiseCorrection') && parameters.noiseCorrection
        phi = phi - 0.2;
    end
    absG = numeratorAbsG./denominatorAbsG; %[Pa]
end
