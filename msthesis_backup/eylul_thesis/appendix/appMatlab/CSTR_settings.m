%%=========================================================================
%%  Settings file for 'U_permeationModelCSTRcomplexRR'
%   takes old simulation and sets last step as inital conditions
%   sets constants
%   sets geometry
%   sets time points
%%=========================================================================

%% time array
ntPointsFull=fix((t_end-t0)/dt)+1;
% t_pointsFull=t0:dt:t_end;   %[s]
% t_points=zeros(1,fix(length(t_pointsFull)/tWriteStep)+2);
t_points(1)=t0;

%% Natural constants
R=8.3147; %J/mol/K, from Peters, Rogg [1993]
NA=6.02214e23; %Avogadro constant

%% constants
kcalTokJ=4.184; % 1kcal=4.184kJ
barToPa=1e5;
atmToBar=1.01325;
m2ToCm2=1e4;
qm2qcm=1e2*1e2*1e2;
m3ToMl=1e6;
T0=273.15; %[k], standard temperature
p0atm=1; %[atm]
p0=p0atm*atmToBar*barToPa; %[Pa]

%% parameters

%geometry
A=62.8e-4/4; % [m^2], Wang 1995 (not divided by 2)
VI=15.7e-6/4;% [m^3], Wang 1995
VII=15.7e-6/4; % [m^3], Wang 1995
L=3.99e-1; % [cm], 1 mm, membrane thickness
% L=0.5e-4; % [cm], 0.5um, membrane thickness, Wang [1995]
CRRthickness=1; % [cm->um] since Wang gives mol/s/qcm, CRR needs a factor from cm to actual reaction thickness  

%settings Wang [1995] reactor volume (prob both): 15.7 cm3 (-> 15.7e-6m3), membrane thickness L=0.5e-6m
%T750°C, A=62.8 cm3 (-> 62.8e-4m2), pressure (both tanks) p=1atm,
%methane flow rate 5cm3/s (changes), air 25cm3/s

%% ======================SET INITIAL CONDITIONS============================
%fixed parameters
PO2I0=0.21*p0atm;      %[atm], literature air: PO2=0.20946*P; for ideal gas Vol-% = partial pressure since pV=nRT
PN2I0=0.79*p0atm;      %[atm], literature air: PN2=0.78084*P, PAr=0.009340*P, other 0.0004338 (e.g. CO2, Ne, He, CH4)
PO2II0=3.06e-4*p0atm;  %[atm]
% PO2II0=0;
PCH40=0.1*p0atm;  %[atm]
PN2II0=1-PCH40-PO2II0; %[atm]
% PN2II0=0;
%fixed value
massFluxIin = 0.03217805*10; %[g/s] if PO2=0.21, N=0.79 -> 25qcm/s @T=273.15
% massFluxIIin = 0.0035797; %[g/s], if PCH4=1-> 5qcm/s @T=273.15
massFluxIIin = 0.0059821; %[g/s], if PCH4=0.1-> 5qcm/s @T=273.15
molarFluxIin = massFluxIin/(PO2I0/p0atm*MW('O2')+PN2I0/p0atm*MW('N2'));
molarFluxIIin = massFluxIIin/(PO2II0*MW('O2')+PCH40*MW('CH4')+PN2II0*MW('N2'));
volFluxIin = molarFluxIin*R.*T_points/p0*qm2qcm; % [cm3]
volFluxIIin = molarFluxIIin*R.*T_points/p0*qm2qcm; % [cm3]
volFluxIinSTD = molarFluxIin*R.*T0/p0*qm2qcm; % [cm3]
volFluxIIinSTD = molarFluxIIin*R.*T0/p0*qm2qcm; % [cm3]
avgResidenceTimeI=VI/volFluxIin*qm2qcm;
avgResidenceTimeII=VII/volFluxIIin*qm2qcm;
N0I = p0*VI/R./T_points; %all moles in each side of the CSTR
N0II = p0*VII/R./T_points; %all moles in each side of the CSTR
m0I=N0I*(PO2I0/p0atm*MW('O2')+PN2I0/p0atm*MW('N2')); %[g] initial mass in whole tank I volume in grams
m0II=N0II*(PO2II0/p0atm*MW('O2')+PCH40/p0atm*MW('CH4')+PN2II0/p0atm*MW('N2')); %[g]

%% Properties for permoeation Models by Xu/Thomson; LSCF-6428==============
% material properties are only dependent on T
Dv0_Xu=1.58e-2;
Dv_Xu=Dv0_Xu*exp(-73.6e3/R./T_points); %[cm^2 s^-1] Data from Xu and Thomson (1999)
Dv2L=Dv_Xu/2/L; %[cm s^-1}
kf_Xu=5.90e6*exp(-226.9e3/R./T_points); %[cm atm^-0.5 s^-1]
kr_Xu=2.07e4*exp(-241.3e3/R./T_points); %[mol cm^-1 s^-1]

%% Intitialize variables (set initial values and create maps of amount of moles of each species
%  Two parameters are used: temperature & time, saved in T x t - matrix

CvI_=zeros(length(T_points),length(t_points));   % O2 vacancy concentration on feed side
CvII_=zeros(length(T_points),length(t_points));  % O2 vacancy concentration on sweep side
JnO2_=zeros(length(T_points),length(t_points));  % [mol/cm^2/s] O2 flux through membrane
JnO2A_=zeros(length(T_points),length(t_points));  % [g/s] O2 flux through membrane
JnCH4A_=zeros(length(T_points),length(t_points)); %
NI_=zeros(length(specieI),length(T_points),length(t_points));   %moles in feed tank
NII_=zeros(length(specieII),length(T_points),length(t_points)); %moles in sweep tank
NIall_=zeros(length(T_points),length(t_points));    %all moles in feed tank
NIIall_=zeros(length(T_points),length(t_points));   %all moles in sweep tank
PI_=zeros(length(specieI),length(T_points),length(t_points));   %partial pressures in feed gas phase
PII_=zeros(length(specieII),length(T_points),length(t_points)); %partial pressures in sweep gas phase
RRII_=zeros(length(specieII),length(T_points),length(t_points)); %Reaction rates at sweep side

tmpCvI_=zeros(1,orderScheme+1);   % O2 vacancy concentration on feed side
tmpCvII_=zeros(1,orderScheme+1);  % O2 vacancy concentration on sweep side
tmpJnO2_=zeros(1,orderScheme+1);  % [mol/cm^2/s] O2 flux through membrane
tmpJnO2A_=zeros(1,orderScheme+1);  % [g/s] O2 flux through membrane
tmpJnO2Ared_=zeros(1,orderScheme+1); % O2 flux that goes in gas phase after catalytic reaction
tmpJnCH4A_=zeros(1,orderScheme+1); %
tmpNI_=zeros(length(specieI),orderScheme+1);   %moles in feed tank
tmpNII_=zeros(length(specieII),orderScheme+1); %moles in sweep tank
tmpNIall_=zeros(1,orderScheme+1);    %all moles in feed tank
tmpNIIall_=zeros(1,orderScheme+1);   %all moles in sweep tank
tmpPI_=zeros(length(specieI),orderScheme+1);   %partial pressures in feed gas phase
tmpPII_=zeros(length(specieII),orderScheme+1); %partial pressures in sweep gas phase
tmpRRII_=zeros(length(specieII),orderScheme+1); %Reaction rates at sweep side

% Set initial partial pressures and total moles of species k in Volume
if(t0==0)
    for(k=1:length(specieI))
        if(strcmp(char(specieI{k}),'O2'))
            PI_(k,:,1)=PO2I0;
            NI_(k,:,1)=N0I*PO2I0;
        elseif(strcmp(char(specieI{k}),'N2'))
            PI_(k,:,1)=PN2I0;
            NI_(k,:,1)=N0I*PN2I0;
        end
    end
    for(k=1:length(specieII))
        if(strcmp(char(specieII{k}),'O2')) 
            PII_(k,:,1)=PO2II0;
            NII_(k,:,1)=N0II*PO2II0;
        elseif(strcmp(char(specieII{k}),'N2'))
            PII_(k,:,1)=PN2II0;
            NII_(k,:,1)=N0II*PN2II0;
        elseif(strcmp(char(specieII{k}),'CH4'))
            PII_(k,:,1)=PCH40;
            NII_(k,:,1)=N0II*PCH40;
        end
    end
% implement previous simulation-
else
    NI_(:,:,1)=finalNI_;
    NII_(:,:,1)=finalNII_;
    PI_(:,:,1)=finalPI_;
    PII_(:,:,1)=finalPII_;
end
%-----------calculate overall initial amount of moles----------------------
for(i=1:length(T_points))
    NIall_(i,1)=sum(NI_(:,i,1)); %calculate total amount of moles in volume I
    NIIall_(i,1)=sum(NII_(:,i,1)); %calculate all moles in volume II
end
