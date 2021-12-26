%%=========================================================================
%%  Research of New Model
%   This script uses different variables to check the following oxygen
%   permeation model through a membrane
%   - D_v*C_v << D_p*C_p and C_p = C0_p (Vacancy diffusion limiting)
%   - 1: CvII*pre_CvII - CRR = pre_kr*kr
%   - 2: CvI=(D_v/2/L*C_v(II)+k_r)/(D_v/2/L+k_f*P_O2^0.5)
%   - 3: J_O2 = D_v/2/L (C_v(II)-C_v(I))
%%=========================================================================

%% variables
clear all;
format long;
syms J kCH4 kOdes KO2s x y;

%% Natural constants
R=8.314; %J/mol/K
NA=6.02214e23; %Avogadro constant

%% constants
kcalTokJ=4.184; % 1kcal=4.184kJ
barToPa=1e5;
atmToBar=1.01325;
m3ToMl=1e6;
T0=273.15; %[k], standard temperature
p0=1e5; %[Pa]

%% material constants
Ea_CH4=14.1e3*kcalTokJ; % [J/mol], data given in 25 kcal/mol comparable to Arai, 1986
k0_CH4=1.43e4;  %mol/s comparable to Arai, 1986

%% parameters
%pressures
PO2I0=0.1; %[atm]
clear PO2I_points;
PO2I_points=[0.21]; %[atm]
% PO2II_points=[3.06e-4 1.00e-5 1.00e-6];
PO2II_points=[3.06e-4 1.00e-6 0];
PCH4_points=[0 0.1]; %[atm]

%temperatures
T_low=873; %[K]
T_high=1223; %[K]
nGrid=3;
T_intervall=(T_high-T_low)/(nGrid-1);
T_points=T_low:T_intervall:T_high;
L=3.99e-1; % [cm], 1 mm

%% Properties for Models 2, 3 & Xu/Thomson; Xu used: LSCF-6428===================
% material used: LSCF-6428
Dv0_Xu=1.58e-2;
Dv_Xu=Dv0_Xu*exp(-73.6e3/R./T_points); %[cm^2 s^-1] Data from Xu and Thomson (1999)
Dv2L=Dv_Xu/2/L; %[cm s^-1}
kf_Xu=5.90e6*exp(-226.9e3/R./T_points); %[cm atm^-0.5 s^-1]
kr_Xu=2.07e4*exp(-241.3e3/R./T_points); %[mol cm^-1 s^-1]

%% Model 1: CRR term! 
% material by Xu LSCF-6428
syms kf kr CvII;

% CRR model
% comparable material according Wang [1995] 
kf01_CH4=10.1;  %mol/s comparable to Arai, 1986
Eaf1_CH4=97e3; % comparable to Arai, 1986
kf1_CH4=kf01_CH4*exp(-Eaf1_CH4/R./T_points);
CRR = @(kfCH4,PCH4,CvII)kfCH4*PCH4; %[mol s^-1 cm^-2]

% initialization
Cv1_I=zeros(length(PO2I_points),length(PO2II_points),length(PCH4_points),length(T_points)); %a for array
Cv1_II=zeros(length(PO2I_points),length(PO2II_points),length(PCH4_points),length(T_points));
J3_nO2=zeros(length(PO2I_points),length(PO2II_points),length(PCH4_points),length(T_points));
J3_vO2=zeros(length(PO2I_points),length(PO2II_points),length(PCH4_points),length(T_points));

%PO2I loop
for(l=1:length(PO2I_points))
    PO2I=PO2I_points(l);
    k=1;

    %PO2II loop
    for(k=1:length(PO2II_points))
        PO2II=PO2II_points(k);
        j=1;

        %PCH4 loop
        for(j=1:length(PCH4_points))
            PCH4=PCH4_points(j);
            i=1;
            
            %Temperature loop model 1
            for(i=1:length(T_points))
                T=T_points(i);
                pre_CvII=(Dv2L(i)*kf_Xu(i)*PO2I^(0.5))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5))+kf_Xu(i)*PO2II^(0.5);
                pre_kr=2-(kf_Xu(i)*PO2I^(0.5))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5));
                Cv1II=solve(CvII*pre_CvII-...
                    CRR(kf1_CH4(i),PCH4,CvII)==...
                    pre_kr*kr_Xu(i),...
                    CvII); %[mol cm^-2]
                Cv1_II(l,k,j,i)= double(Cv1II);
                Cv1_I(l,k,j,i)=(Dv2L(i)*Cv1_II(l,k,j,i)+kr_Xu(i))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5));
                J1_nO2(l,k,j,i)=Dv2L(i)*(Cv1_II(l,k,j,i)-Cv1_I(l,k,j,i)); %[mol cm^-2 s^1]
                J1_vO2(l,k,j,i)=J1_nO2(l,k,j,i)*R*T0*m3ToMl*60/p0; %pV=nRT, Xu (1999) used standard conditions in order to normalize
                %CRR31(l,k,j,i)=CRR(kf3_CH4(i),PCH4,Cv1II);
            end;
            clear PCH4;
        end;
        clear PO2II;
    end;
    clear PO2I;
end;

%% plot Model 1 results
figure
box on;
nPlots=1;
for(l=1:length(PO2I_points))
    for(k=1:length(PO2II_points))
        for(j=1:length(PCH4_points))
            for(i=1:length(T_points))
                J1_temp(i)=J1_vO2(l,k,j,i);
%                 CvII_temp(i)=Cv1_II(l,k,j,i);
            end;
            if(j==1)
               semilogy(T_points-273,J1_temp,'x-');
               legendInfo{nPlots} = strcat('J(P_{CH4} = ',num2str(PCH4_points(j)),' atm, P^{"}_{O2} = ',num2str(PO2II_points(k)),' atm)');
               nPlots=nPlots+1;
               hold on
            end
            if(j==2)
               semilogy(T_points-273,J1_temp,'o-');
               legendInfo{nPlots} = strcat('J(P_{CH4} = ',num2str(PCH4_points(j)),' atm, P^{"}_{O2} = ',num2str(PO2II_points(k)),' atm)');
               nPlots=nPlots+1;
               hold on
            end
        end
    end
end;
xlabel('temperature, T [°C]');
ylabel('oxygen flux, J_{O_2} [ml cm^{-2} min^{-1}]');
legend(legendInfo);
hold off;