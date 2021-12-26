%%=========================================================================
%%  Comarison of kinetics of oxidation of CH3 by O2 to CO2 annd H2O
%   
%   ANNOTATION:
%   most relevant oxidation path is taken using the path by Warnatz (1984)
%   and the thermophysical data by Bilger (1990)
%   overall: 21 reactions with 12 species + CH4
%%=========================================================================

clear all;

%% constants
calToJ=4.184; % 1kcal=4.184kJ
R=8.314;

%% Properties

%thermophysical data
reactions={'CH3,O','CH3,OH','CH2O,OH','CH2O,H','CH2O,O',...
    'HCO,OH','HCO,H','HCO,O','HCO,O2','CO,O','CO,OH','CO,O2','CO,HO2',...
    'H2,O2','H2,OH','O2,H','O,H2','2OH'};

A=[6.8e13 1e12 3.43e9 2.19e8 1.81e13...
    5.0e12 4.0e13 1e13 3.3e13 3.2e13 1.51e7 1.6e13 5.0e13...
    1.7e13 1.17e9 5.13e16 1.8e10 6e8];
b=[0 0 1.18 1.77 0 ...
    0 0 0 -0.4 0 1.3 0 ...
    0 0 1.3 -0.816 1.0 1.3];
Ea=[0 0 -447 3000 3082 ...
    0 0 0 0 -4200 -758 46000 ...
    22934 47780 3626 16507 8826 0];

%temperatures
T_low=673.15; %[K]
T_high=2273; %[K]
nGridT=15; %needs to be at least 2
T_intervall=(T_high-T_low)/(nGridT-1);
T_points=T_low:T_intervall:T_high;

%% initialize
% k=ones(length(T_points),length(reactions));

%% Arrhenius rates
for(i=1:length(T_points))
    for(j=1:length(reactions))
        k(i,j) = A(j)*T_points(i)^b(j)*exp(-Ea(j)*calToJ/R/T_points(i));
    end
end
kCH4=6.3e14*exp(-104000*calToJ/R./T_points);

%% plot
fig=figure;
box on;
semilogy(T_points-273,k(:,1));
legendEntries{1}=strcat(reactions{1});
xlabel('Temperature, T in [°C]')
ylabel('reaction rate, k in [mol cm^{-3} s^{-1}]')
hold on;
for(j=2:length(reactions))
    semilogy(T_points-273,k(:,j));
    legendEntries{j}=strcat(reactions{j});
end
semilogy(T_points-273,kCH4);
legendEntries{j+1}='CH4';
legend(legendEntries);
hold off;

%% plot 1/T
fig=figure;
box on;
semilogy(1./(T_points-273),k(:,1));
legendEntries{1}=strcat(reactions{1});
xlabel('inverse temperature, 1/T in [°C^{-1}]')
ylabel('reaction rate, k in [mol cm^{-3} s^{-1}]')
hold on;
for(j=2:length(reactions))
    semilogy(1./(T_points-273),k(:,j));
    legendEntries{j}=strcat(reactions{j});
end
semilogy(1./(T_points-273),kCH4);
legendEntries{j+1}='CH4';
legend(legendEntries);
hold off;
