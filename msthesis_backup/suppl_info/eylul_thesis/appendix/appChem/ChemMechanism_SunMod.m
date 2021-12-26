%%=========================================================================
%%  Gas phase chemistry file for 'U_permeationModelCSTRcomplexRR'
%   Mechanism based on "Microkinetics of methane oxidative coupling" by
%	Sun et al., 2008
%	k0 has units (s-1 or m3 mol-1 s-1 or m6 mol-2 s-1)
%	Ea has units (kJ mol-1)
%--------------------------------------------------------------------------
%   Description:
%   - if a function is a three body reaction it needs the signal word
%     '3B';
%   - if function is a fall-off reaction:reacs(n).reacType='TROE' or 'TROELin'
%%=========================================================================

calToJ=4.184; % 1cal=4.184J
J2kJ=1e3;

%% ========================SPECIE & MOLAR WEIGHT===========================
specieI={'O2','N2'}; %ATTENTION: DO NOT CHANGE ORDER!!! {'O2','N2'}
specieII={'O2','N2','CH4','CH3','H2O','CO2',...
    'H','H2','HO2','H2O2','OH','O','CO',...
	'CHO','CH2O','CH3O',...
	'C2H2','C2H3','C2H4','C2H5','C2H6'}; %ATTENTION: ALWAYS KEEP ORDER of specieII and specieMW the same!!!
	
specieMW=[31.99880,28.01340,16.0425,15.0345,18.01528,44.0095,...
	1.00794,2.01588,33.00674,34.0147,17.008,15.9994,28.0101,...
	29.0186,30.0260,31.0339,...
	26.0373,27.0452,28.0532,29.0611,30.0690];
	
MW=containers.Map(specieII,specieMW); %[g/mol]


%% ========================CHEMICAL REACTIONS==============================
reacs=ClassReaction;
n=0;
%------------------------CH4 consumption-----------------------------------
n=n+1;  reacs(n).educts={'CH4','O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3','HO2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.983e7; reacs(n).b=0; reacs(n).Ea=193.86*J2kJ;
n=n+1;  reacs(n).educts={'CH4','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3','H2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.234e9; reacs(n).b=0; reacs(n).Ea=51.17*J2kJ;
n=n+1;  reacs(n).educts={'CH4','OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3','H2O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.127e10; reacs(n).b=0; reacs(n).Ea=33.83*J2kJ;
n=n+1;  reacs(n).educts={'CH4','OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3','H2O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.743e9; reacs(n).b=0; reacs(n).Ea=41.43*J2kJ;
n=n+1;  reacs(n).educts={'CH4','HO2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3','H2O2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.401e8; reacs(n).b=0; reacs(n).Ea=99.61*J2kJ;
%------------------------CH3 consumption-----------------------------------
n=n+1;  reacs(n).educts={'CH3','O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3O','O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.308e9; reacs(n).b=0; reacs(n).Ea=141.00*J2kJ;
n=n+1;  reacs(n).educts={'CH3','O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH2O','OH'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.459e8; reacs(n).b=0; reacs(n).Ea=103.66*J2kJ;		
n=n+1;  reacs(n).educts={'CH3','HO2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3O','OH'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.885e8; reacs(n).b=0; reacs(n).Ea=0;		
n=n+1;  reacs(n).educts={'CH3','CH3','M'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H6','M'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.650e8; reacs(n).b=0; reacs(n).Ea=0;
		reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4' 'CO2' 'CO' 'N2' 'H2O' 'O2' 'other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
%------------------------CH3O consumption----------------------------------
n=n+1;  reacs(n).educts={'CH3O','M'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH2O','H','M'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.258e15; reacs(n).b=0; reacs(n).Ea=115.00*J2kJ;
		reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4' 'CO2' 'CO' 'N2' 'H2O' 'O2' 'other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
%------------------------CH2O consumption----------------------------------		
n=n+1;  reacs(n).educts={'CH2O','OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CHO','H2O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.580e9; reacs(n).b=0; reacs(n).Ea=5.00*J2kJ;
n=n+1;  reacs(n).educts={'CH2O','HO2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CHO','H2O2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.417e7; reacs(n).b=0; reacs(n).Ea=40.12*J2kJ;
n=n+1;  reacs(n).educts={'CH2O','CH3'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CHO','CH4'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.700e8; reacs(n).b=0; reacs(n).Ea=25.03*J2kJ;
%------------------------CHO consumption-----------------------------------			
n=n+1;  reacs(n).educts={'CHO','M'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CO','H','M'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.280e10; reacs(n).b=0; reacs(n).Ea=64.36*J2kJ;
		reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
n=n+1;  reacs(n).educts={'CHO','O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CO','HO2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.171e6; reacs(n).b=0; reacs(n).Ea=0*J2kJ;
%------------------------CO consumption------------------------------------			
n=n+1;  reacs(n).educts={'CO','HO2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CO2','OH'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.308e9; reacs(n).b=0; reacs(n).Ea=107.34*J2kJ;
%------------------------C2H6 consumption----------------------------------			
n=n+1;  reacs(n).educts={'C2H6','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H5','H2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.910e9; reacs(n).b=0; reacs(n).Ea=51.70*J2kJ;
n=n+1;  reacs(n).educts={'C2H6','OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H5','H2O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.545e9; reacs(n).b=0; reacs(n).Ea=17.16*J2kJ;
n=n+1;  reacs(n).educts={'C2H6','CH3'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H5','CH4'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.239e9; reacs(n).b=0; reacs(n).Ea=64.73*J2kJ;
n=n+1;  reacs(n).educts={'C2H6'}; reacs(n).stoiEducts=[1];
        reacs(n).products={'C2H5','H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.400e17; reacs(n).b=0; reacs(n).Ea=378.51*J2kJ;
%------------------------C2H5 consumption----------------------------------			
n=n+1;  reacs(n).educts={'C2H5','HO2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3','CH2O','OH'}; reacs(n).stoiProducts=[1 1 1];
        reacs(n).k0=0.948e7; reacs(n).b=0; reacs(n).Ea=0;
n=n+1;  reacs(n).educts={'C2H5','M'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H4','H','M'}; reacs(n).stoiProducts=[1 1 1];
        reacs(n).k0=0.596e14; reacs(n).b=0; reacs(n).Ea=167.66*J2kJ;
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);		
n=n+1;  reacs(n).educts={'C2H5','O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H4','HO2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.635e7; reacs(n).b=0; reacs(n).Ea=53.20*J2kJ;
%------------------------C2H4 consumption----------------------------------			
n=n+1;  reacs(n).educts={'C2H4','O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H3','HO2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.281e7; reacs(n).b=0; reacs(n).Ea=144.55*J2kJ;
n=n+1;  reacs(n).educts={'C2H4','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H3','H2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.150e9; reacs(n).b=0; reacs(n).Ea=42.70*J2kJ;
n=n+1;  reacs(n).educts={'C2H4','OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H3','H2O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.612e8; reacs(n).b=0; reacs(n).Ea=24.70*J2kJ;
n=n+1;  reacs(n).educts={'C2H4','CH3'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H3','CH4'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.199e6; reacs(n).b=0; reacs(n).Ea=51.46*J2kJ;
n=n+1;  reacs(n).educts={'C2H4','OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH3','CH2O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.272e7; reacs(n).b=0; reacs(n).Ea=0.00*J2kJ;
%------------------------C2H3 consumption----------------------------------			
n=n+1;  reacs(n).educts={'C2H3','M'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H','H','M'}; reacs(n).stoiProducts=[1 1 1];
        reacs(n).k0=0.121e16; reacs(n).b=0; reacs(n).Ea=176.44*J2kJ;
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);		
n=n+1;  reacs(n).educts={'C2H3','O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'C2H3','HO2'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.500e7; reacs(n).b=0; reacs(n).Ea=0;
n=n+1;  reacs(n).educts={'C2H3','O2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'CH2O','CHO'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.550e7; reacs(n).b=0; reacs(n).Ea=0;
%------------------------no C3-species!------------------------------------
%------------------------H/H2/HO2/H2O2/OH/O/O2-----------------------------		
n=n+1;  reacs(n).educts={'O2','H'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'OH','O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.220e9; reacs(n).b=0; reacs(n).Ea=70.30*J2kJ;
n=n+1;  reacs(n).educts={'O2','H','M'}; reacs(n).stoiEducts=[1 1 1];
        reacs(n).products={'HO2','M'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=0.139e6; reacs(n).b=0; reacs(n).Ea=0;
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
n=n+1;  reacs(n).educts={'HO2','HO2'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'O2','OH','OH'}; reacs(n).stoiProducts=[1 1 1];
        reacs(n).k0=0.200e7; reacs(n).b=0; reacs(n).Ea=70;
n=n+1;  reacs(n).educts={'H2O2','M'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'OH','OH','M'}; reacs(n).stoiProducts=[1 1 1];
        reacs(n).k0=0.127e12; reacs(n).b=0; reacs(n).Ea=199.36*J2kJ;
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
%------------------------H2 break up/OH from Peters&Rogg (1993)------------
n=n+1;  reacs(n).educts={'H2','O'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'OH','H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=5.060e-2; reacs(n).b=2.67; reacs(n).Ea=26.3*J2kJ; %m3,mole,sec
n=n+1;  reacs(n).educts={'H2','OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'H2O','H'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.00e2; reacs(n).b=1.6; reacs(n).Ea=13.8*J2kJ; %m3,mole,sec
n=n+1;  reacs(n).educts={'OH','OH'}; reacs(n).stoiEducts=[1 1];
        reacs(n).products={'H2O','O'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.9e11; reacs(n).b=-1; reacs(n).Ea=0; %m3,mole,sec
%------------------------Recombination from Peters&Rogg (1993)-------------	       
n=n+1;  reacs(n).educts={'H','H','M'}; reacs(n).stoiEducts=[1 1 1];
        reacs(n).products={'H2','M'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=1.8e12; reacs(n).b=-1; reacs(n).Ea=0; %m3,mole,sec
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
n=n+1;  reacs(n).educts={'OH','H','M'}; reacs(n).stoiEducts=[1 1 1];
        reacs(n).products={'H2O','M'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.2e16; reacs(n).b=-2; reacs(n).Ea=0; %m3,mole,sec
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
n=n+1;  reacs(n).educts={'O','O','M'}; reacs(n).stoiEducts=[1 1 1];
        reacs(n).products={'O2','M'}; reacs(n).stoiProducts=[1 1];
        reacs(n).k0=2.9e11; reacs(n).b=-1; reacs(n).Ea=0; %m3,mole,sec
        reacs(n).reacType='3B';
        reacs(n).thirdBodyMap=containers.Map(...
            {'CH4','CO2','CO','N2','H2O','O2','other'},...
            [6.5 1.5 0.75 0.4 6.5 0.4 1.0]);
%--------------------------------------------------------------------------
clear n;
