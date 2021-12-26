%%=========================================================================
%%  Gas phase chemistry file for 'U_permeationModelCSTRcomplexRR'
%   Mechanism copied from 'Reduced Kinetic Mechanisms for Applications in
%   Combustion Systems', 1993, by Peters and Rogg
%   CH4 break up taken from "Microkinetics of methane oxidative coupling" 
%	by Sun et al., 2008
%	k0 has units (s-1 or m3 mol-1 s-1 or m6 mol-2 s-1)
%	Ea has units (kJ mol-1)
%--------------------------------------------------------------------------
%   Description:
%   - if a function is a three body reaction it needs the signal word
%     '3B';
%   - if function is a fall-off reaction:reacs(n).reacType='TROE' or 'TROELin'
%     (to see influence look into ClassReaction.m)
%%=========================================================================

%%conversion constants
cal2J=4.184; % 1cal=4.184J
J2kJ=1e3;

%% ========================SPECIE & MOLAR WEIGHT===========================
specieI={'O2','N2'}; %ATTENTION: DO NOT CHANGE ORDER!!! {'O2','N2'}
specieII={'O2','N2','CH4','CH3','H2O','CO2',...
    'H','H2','HO2','H2O2','OH','O','CO',...
	'CH','CH2','CHO','CH2O','CH3O',...
	'C2HO','C2H','C2H2','C2H3','C2H4','C2H5','C2H6'}; %ATTENTION: ALWAYS KEEP ORDER of specieII and specieMW the same!!!
	
specieMW=[31.99880,28.01340,16.0425,15.0345,18.01528,44.0095,...
	1.00794,2.01588,33.00674,34.0147,17.008,15.9994,28.0101,...
	13.01864,14.02658,29.0186,30.0260,31.0339,...
	41.0287,25.0293,26.0373,27.0452,28.0532,29.0611,30.0690];
	
MW=containers.Map(specieII,specieMW); %[g/mol]


%% ========================CHEMICAL REACTIONS==============================
reacs=ClassReaction;
n=0;
%------------------------H2/O2 Chain Reactions-----------------------------
n=n+1;  reacs(n).educts={'O2','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'OH','O'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.000E+14; reacs(n).b=0; reacs(n).Ea=70.30e3;
n=n+1;  reacs(n).educts={'OH' 'O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'O2' 'H'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.568E+13; reacs(n).b=0; reacs(n).Ea=3.52e3;
        %forward/backward------------------------------------------
n=n+1;  reacs(n).educts={'H2' 'O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'OH' 'H'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=5.060E+04; reacs(n).b=2.67; reacs(n).Ea=26.30e3;
n=n+1;  reacs(n).educts={'OH' 'H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'H2' 'O'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.222E+04; reacs(n).b=2.67; reacs(n).Ea=18.29e3;
        %forward/backward------------------------------------------
n=n+1;  reacs(n).educts={'H2' 'OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'H2O' 'H'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.000E+08; reacs(n).b=1.60; reacs(n).Ea=13.80e3;
n=n+1;  reacs(n).educts={'H2O' 'H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'H2' 'OH'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=4.312E+08; reacs(n).b=1.60; reacs(n).Ea=76.46e3;
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'OH'}; reacs(n).stoiEducts=[2];
        reacs(n).products={'H2O','O'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.500E+09; reacs(n).b=1.14; reacs(n).Ea=0.42e3;
n=n+1;  reacs(n).educts={'H2O','O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'OH'};  reacs(n).stoiProducts=[2];
        reacs(n).k0=1.473E+10; reacs(n).b=1.14; reacs(n).Ea=71.09e3;
%------------------------HO2 Formation and Consumption---------------------
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'O2','H','M'}; reacs(n).stoiEducts=[1 1 1];
        reacs(n).products={'HO2','M'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.300E+18; reacs(n).b=-0.80; reacs(n).Ea=0.00;        
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4' 'CO2' 'CO' 'N2' 'H2O' 'O2' 'other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
n=n+1;  reacs(n).educts={'HO2','M'};  reacs(n).stoiEducts=[1 1];
        reacs(n).products={'O2','H','M'}; reacs(n).stoiProducts=[1 1 1];
        reacs(n).k0=3.190E+18; reacs(n).b=-0.80; reacs(n).Ea=195.39e3;        
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4' 'CO2' 'CO' 'N2' 'H2O' 'O2' 'other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
        %----------------------------------------------------------
n=n+1;  reacs(n).educts={'HO2','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'OH'};  reacs(n).stoiProducts=[2];
        reacs(n).k0=1.500E+14; reacs(n).b=0.00; reacs(n).Ea=4.20e3;
n=n+1;  reacs(n).educts={'HO2','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'H2','O2'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.500E+13; reacs(n).b=0.00; reacs(n).Ea=2.90e3;
n=n+1;  reacs(n).educts={'HO2','OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'H2O','O2'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=6.000E+13; reacs(n).b=0.00; reacs(n).Ea=0.00;
n=n+1;  reacs(n).educts={'HO2','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'H2O','O'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.000E+13; reacs(n).b=0.00; reacs(n).Ea=7.20e3;
n=n+1;  reacs(n).educts={'HO2','O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'OH','O2'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.800E+13; reacs(n).b=0.00; reacs(n).Ea=-1.70e3;
%------------------------H2O2 Formation and Consumption--------------------
n=n+1;  reacs(n).educts={'HO2','HO2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'H2O2','O2'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.500E+11; reacs(n).b=0.00; reacs(n).Ea=-5.20e3;
        reacs(n).uniqueKey='11';
        %forward/backward-------------------------------------------
% n=n+1;  reacs(n).educts={'OH','M'}; reacs(n).stoiEducts=[2 1];
%         reacs(n).products={'H2O2','M'};  reacs(n).stoiProducts=[1 1];
%         reacs(n).k0=3.250E+22; reacs(n).b=-2.00; reacs(n).Ea=0.00;
%         reacs(n).reacType='3B';
%         reacs(n).thirdBodyMap=containers.Map(...
%             {'CH4' 'CO2' 'CO' 'N2' 'H2O' 'O2' 'other'},...
%             [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
n=n+1;  reacs(n).educts={'H2O2','M'};  reacs(n).stoiEducts=[1 1];
        reacs(n).products={'OH','M'}; reacs(n).stoiProducts=[2 1];
        reacs(n).k0=1.692E+24; reacs(n).b=-2.00; reacs(n).Ea=202.29;
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4' 'CO2' 'CO' 'N2' 'H2O' 'O2' 'other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
        %-----------------------------------------------------------
n=n+1;  reacs(n).educts={'H2O2','H'};  reacs(n).stoiEducts=[1 1];
        reacs(n).products={'H2O','OH'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.000E+13; reacs(n).b=0.00; reacs(n).Ea=15.00e3;
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'H2O2','OH'};  reacs(n).stoiEducts=[1 1];
        reacs(n).products={'H2O','HO2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=5.400E+12; reacs(n).b=0.00; reacs(n).Ea=4.20e3;
n=n+1;  reacs(n).educts={'H2O','HO2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'H2O2','OH'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.802E+13; reacs(n).b=0.00; reacs(n).Ea=134.75e3;
%------------------------Recombination from Peters&Rogg (1993)-------------	       
n=n+1;  reacs(n).educts={'H','H','M'}; reacs(n).stoiEducts=[1 1 1];
        reacs(n).products={'H2','M'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.8e18; reacs(n).b=-1; reacs(n).Ea=0; %m3,mole,sec
        reacs(n).uniqueKey='15';
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
n=n+1;  reacs(n).educts={'OH','H','M'}; reacs(n).stoiEducts=[1 1 1];
        reacs(n).products={'H2O','M'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.2e22; reacs(n).b=-2; reacs(n).Ea=0; %m3,mole,sec
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
n=n+1;  reacs(n).educts={'O','O','M'}; reacs(n).stoiEducts=[1 1 1];
        reacs(n).products={'O2','M'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.9e17; reacs(n).b=-1; reacs(n).Ea=0; %m3,mole,sec
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
%------------------------CO/CO2 Mechanism----------------------------------
n=n+1;  reacs(n).educts={'CO' 'OH'};  reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CO2' 'H'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=4.400E+06; reacs(n).b=1.50; reacs(n).Ea=-3.10e3;
n=n+1;  reacs(n).educts={'CO2' 'H'};  reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CO' 'OH'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=4.956E+08; reacs(n).b=1.50; reacs(n).Ea=89.76e3;
%------------------------CH consumption------------------------------------
n=n+1;  reacs(n).educts={'CH' 'O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CHO' 'O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.000E+13; reacs(n).b=0; reacs(n).Ea=0;
n=n+1;  reacs(n).educts={'CH' 'CO2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CHO' 'CO'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.400E+12; reacs(n).b=0; reacs(n).Ea=2.90e3; 
%------------------------CHO consumption-----------------------------------        
n=n+1;  reacs(n).educts={'CHO' 'H'};  reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CO' 'H2'};  reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.000E+14; reacs(n).b=0; reacs(n).Ea=0;
n=n+1;  reacs(n).educts={'CHO' 'OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CO' 'H2O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.000E+14; reacs(n).b=0; reacs(n).Ea=0;
n=n+1;  reacs(n).educts={'CHO' 'O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'HO2' 'CO'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.000E+12; reacs(n).b=0; reacs(n).Ea=0.00;
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'CHO' 'M'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CO' 'H','M'}; reacs(n).stoiProducts=[1 1 1];
        reacs(n).k0=7.100E+14; reacs(n).b=0; reacs(n).Ea=70.30e3;
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
n=n+1;  reacs(n).educts={'CO' 'H','M'}; reacs(n).stoiEducts=[1 1 1];
        reacs(n).products={'CHO' 'M'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.136E+15; reacs(n).b=0; reacs(n).Ea=9.97e3;
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
%------------------------CH2 consumption-----------------------------------
n=n+1;  reacs(n).educts={'CH2' 'H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH' 'H2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=8.400E+09; reacs(n).b=1.50; reacs(n).Ea=1.40e3;
n=n+1;  reacs(n).educts={'CH' 'H2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH2' 'H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=5.830E+09; reacs(n).b=1.50; reacs(n).Ea=13.08e3;
        %-----------------------------------------------------------
n=n+1;  reacs(n).educts={'CH2' 'O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CO' 'H'}; reacs(n).stoiProducts=[1 2];
        reacs(n).k0=8.000E+13; reacs(n).b=0; reacs(n).Ea=0;
n=n+1;  reacs(n).educts={'CH2' 'O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CO' 'OH' 'H'}; reacs(n).stoiProducts=[1 1 1];
        reacs(n).k0=6.500E+12; reacs(n).b=0; reacs(n).Ea=6.30e3;
n=n+1;  reacs(n).educts={'CH2' 'O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CO2' 'H'}; reacs(n).stoiProducts=[1 2];
        reacs(n).k0=6.500E+12; reacs(n).b=0; reacs(n).Ea=6.30e3;%same as above, no mistake!
%------------------------CH2O consumption----------------------------------
n=n+1;  reacs(n).educts={'CH2O' 'H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CHO' 'H2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.500E+13; reacs(n).b=0; reacs(n).Ea=16.70e3;
n=n+1;  reacs(n).educts={'CH2O' 'O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CHO' 'OH'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.500E+13; reacs(n).b=0; reacs(n).Ea=14.60e3;
n=n+1;  reacs(n).educts={'CH2O' 'OH'};reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CHO' 'H2O'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.000E+13; reacs(n).b=0; reacs(n).Ea=5.00e3;
n=n+1;  reacs(n).educts={'CH2O' 'M'};reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CHO' 'H','M'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.400E+17; reacs(n).b=0; reacs(n).Ea=320.00e3;
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
%------------------------CH3 consumption-----------------------------------
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'CH3' 'H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH2' 'H2'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.800E+14; reacs(n).b=0.00; reacs(n).Ea=63.00e3;
n=n+1;  reacs(n).educts={'CH2' 'H2'};reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3' 'H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.680E+13; reacs(n).b=0.00; reacs(n).Ea=44.30e3;
        %-----------------------------------------------------------
n=n+1;  reacs(n).educts={'CH3' 'H'};reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH4'}; reacs(n).stoiProducts=[1];
        reacs(n).reacType='TROELin';
        reacs(n).k0=[2.108E+14 6.257E+23];
        reacs(n).b=[0.00 -1.80];
        reacs(n).Ea=[0 0];
        reacs(n).uniqueKey='34';
        %-----------------------------------------------------------
n=n+1;  reacs(n).educts={'CH3' 'O'};  reacs(n).stoiEducts=[1 1]; %main path
        reacs(n).products={'CH2O' 'H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=7.000E+13; reacs(n).b=0; reacs(n).Ea=0;
        %-----------------------------------------------------------
n=n+1;  reacs(n).educts={'CH3'};reacs(n).stoiEducts=[2];
        reacs(n).products={'C2H6'}; reacs(n).stoiProducts=[1];
        reacs(n).reacType='TROELin';
        reacs(n).k0=[3.613E+13 1.270E+41];
        reacs(n).b=[0.00 -7.00];
        reacs(n).Ea=[0 11.56e3];
        reacs(n).uniqueKey='36';
        %-----------------------------------------------------------
n=n+1;  reacs(n).educts={'CH3' 'O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH2O' 'OH'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.400E+11; reacs(n).b=0; reacs(n).Ea=37.40e3;
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'CH4' 'H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3' 'H2'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.200E+04; reacs(n).b=3.00; reacs(n).Ea=36.60e3;
n=n+1;  reacs(n).educts={'CH3' 'H2'};reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH4' 'H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=8.391E+02; reacs(n).b=3.00; reacs(n).Ea=34.56e3;
        %-----------------------------------------------------------
n=n+1;  reacs(n).educts={'CH4' 'O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3' 'OH'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.200E+07; reacs(n).b=2.10; reacs(n).Ea=31.90e3;
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'CH4' 'OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3' 'H1O'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.600E+06; reacs(n).b=2.10; reacs(n).Ea=10.30e3;
n=n+1;  reacs(n).educts={'CH3' 'H2O'};reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH4' 'OH'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.631E+05; reacs(n).b=2.10; reacs(n).Ea=70.92e3;
        %add equation for CH4 break-up with O2 (Sun, 2008)---------
n=n+1;  reacs(n).educts={'CH4' 'O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3' 'HO2'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.983e13; reacs(n).b=0; reacs(n).Ea=193.86e3;
%========================C2HO consumption==================================
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'C2HO' 'H'};reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH2' 'CO'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.000E+13; reacs(n).b=0.00; reacs(n).Ea=0;
n=n+1;  reacs(n).educts={'CH2' 'CO'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2HO' 'H'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.361E+12; reacs(n).b=0.00; reacs(n).Ea=-29.39e3;
        %-----------------------------------------------------------
n=n+1;  reacs(n).educts={'C2HO' 'OH'};reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CO' 'H'}; reacs(n).stoiProducts=[2 1];
        reacs(n).k0=1.000E+14; reacs(n).b=0.00; reacs(n).Ea=0.00;
%========================C2H consumption===================================
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'C2H' 'H2'};reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H2' 'H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.100E+13; reacs(n).b=0.00; reacs(n).Ea=12.00e3;
n=n+1;  reacs(n).educts={'C2H2' 'H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2HO' 'O'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=5.270E+13; reacs(n).b=0.00; reacs(n).Ea=119.95e3;
        %-----------------------------------------------------------
n=n+1;  reacs(n).educts={'C2H' 'O2'};reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2HO' 'H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=5.000E+13; reacs(n).b=0.00; reacs(n).Ea=6.30e3;
%------------------------C2H2 consumption----------------------------------
n=n+1;  reacs(n).educts={'C2H2' 'O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH2' 'CO'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=4.100E+08; reacs(n).b=1.50; reacs(n).Ea=7.10e3;
n=n+1;  reacs(n).educts={'C2H2' 'O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2HO' 'H'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=4.300E+14; reacs(n).b=0.00; reacs(n).Ea=50.70e3;
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'C2H2' 'OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H' 'H2O'};reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.000E+13; reacs(n).b=0.00; reacs(n).Ea=29.30e3;
        reacs(n).uniqueKey='47f';
n=n+1;  reacs(n).educts={'C2H' 'H2O'};reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H2' 'OH'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=9.000E+12; reacs(n).b=0.00; reacs(n).Ea=-15.98e3;
        reacs(n).uniqueKey='47b';
        %-----------------------------------------------------------
        %equation 48 excluded because of C3-species
%------------------------C2H3 consumption----------------------------------
n=n+1;  reacs(n).educts={'C2H3','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H2','H2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.000E+13; reacs(n).b=0.00; reacs(n).Ea=0.00;
n=n+1;  reacs(n).educts={'C2H3','O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H2','HO2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=5.400E+11; reacs(n).b=0.00; reacs(n).Ea=0.00;
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'C2H3'};reacs(n).stoiEducts=[1];
        reacs(n).products={'C2H2','H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).reacType='TROELin';
        reacs(n).k0=[2.000E+14 1.187E+42];
        reacs(n).b=[0.00 -7.50];
        reacs(n).Ea=[166.29e3 190.40e3];
n=n+1;  reacs(n).educts={'C2H2','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H3'};reacs(n).stoiProducts=[1];
        reacs(n).reacType='TROELin';
        reacs(n).k0=[1.053E+14 1.187E+42];
        reacs(n).b=[0.00 -7.50];
        reacs(n).Ea=[3.39e3 190.40e3];
%------------------------C2H4 consumption----------------------------------
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'C2H4','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H3','H2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.500E+14; reacs(n).b=0; reacs(n).Ea=42.70e3;
n=n+1;  reacs(n).educts={'C2H3','H2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H4','H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=9.605E+12; reacs(n).b=0.00; reacs(n).Ea=32.64e3;
        %-----------------------------------------------------------
n=n+1;  reacs(n).educts={'C2H4','O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3','CO','H'}; reacs(n).stoiProducts=[1 1 1];
        reacs(n).k0=1.600E+09; reacs(n).b=1.20; reacs(n).Ea=3.10e3;  
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'C2H4','OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H3','H2O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.000E+13; reacs(n).b=0.00; reacs(n).Ea=12.60e3;
n=n+1;  reacs(n).educts={'C2H3','H2O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H4','OH'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=8.283E+12; reacs(n).b=0.00; reacs(n).Ea=65.20e3;
        %-----------------------------------------------------------
n=n+1;  reacs(n).educts={'C2H4','M'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H2','H2','M'}; reacs(n).stoiProducts=[1 1 1];
        reacs(n).k0=2.500E+17; reacs(n).b=0.00; reacs(n).Ea=319.80e3;
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
%------------------------C2H5 consumption----------------------------------
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'C2H5','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3','CH3'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.000E+13; reacs(n).b=0; reacs(n).Ea=0.00;
n=n+1;  reacs(n).educts={'CH3','CH3'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H5','H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.547E+12; reacs(n).b=0; reacs(n).Ea=49.68e3;
        %-----------------------------------------------------------
n=n+1;  reacs(n).educts={'C2H5','O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H4','HO2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.000E+12; reacs(n).b=0; reacs(n).Ea=20.90e3;
        %forward/backward-------------------------------------------
n=n+1;  reacs(n).educts={'C2H5'};reacs(n).stoiEducts=[1];
        reacs(n).products={'C2H4','H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).reacType='TROELin';
        reacs(n).k0=[2.000E+13 1.000E+17];
        reacs(n).b=[0.00 0.00];
        reacs(n).Ea=[166.00e3 130.00e3];
n=n+1;  reacs(n).educts={'C2H4','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H5'};reacs(n).stoiProducts=[1];
        reacs(n).reacType='TROELin';
        reacs(n).k0=[3.189E+13 1.000E+17];
        reacs(n).b=[0.00 0.00];
        reacs(n).Ea=[12.61e3 130.00e3];
%------------------------C2H6 consumption----------------------------------
n=n+1;  reacs(n).educts={'C2H6','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H5','H2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=5.400E+02; reacs(n).b=3.50; reacs(n).Ea=21.80e3;
n=n+1;  reacs(n).educts={'C2H6','O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H5','OH'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=3.000E+07; reacs(n).b=2.00; reacs(n).Ea=21.40e3;
n=n+1;  reacs(n).educts={'C2H6','OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H5','H2O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=6.300E+06; reacs(n).b=2.00; reacs(n).Ea=2.70e3;
%--------------------------------------------------------------------------
clear n;
%change units to m3
for(n=1:length(reacs))
    reacs(n).k0=reacs(n).k0*1e-6;
end