%%=========================================================================
%% Comparison of Permeation Models
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
k0_O2lat=7.74e16; %from Markova-Velichkova, 2013 - MVK-D
k0_Odes=2.07e5; %from Markova-Velichkova, 2013 - MVK-D
Ea_O2lat=172.0e3; % [J/mol], data given in [kJ/mol] from Markova-Velichkova, 2013 - MVK-D
Ea_O2des=40.5e3; % [J/mol], data given in [kJ/mol] from Markova-Velichkova, 2013 - MVK-D

Cv0=1e-4;
Cp0=1e-3;
CT0=2*Cv0+Cp0;
Dp0=1.1e-2; %coefficient for vacancy diffusion; INFO: divided by 2Dv-Dp -> 2Dv < Dp
Dv0=1.1e-2; %coefficient for electron hole;
Ea_Dp=7.56e4;
Ea_Dv=7.56e4;

%% parameters
%pressures
PO2I0=0.1; %[atm]
clear PO2I_points;
PO2I_points=0.21; %[atm]
PO2II_points=[2.96e-3 1.05e-3 3.06e-4];
PCH4_points=[0 0.01 0.03 0.1]; %[atm]

%temperatures
T_low=873; %[K]
T_high=1823; %[K]
nGrid=30;
T_intervall=(T_high-T_low)/(nGrid-1);
T_points=T_low:T_intervall:T_high;
L=3.99e-1; % [cm], 1 mm

%% reaction velocities
k_CH4=k0_CH4*exp(-Ea_CH4/R./T_points); % CH4 + lattice oxygen to CH3 and water
k_O2lat=k0_O2lat*exp(-Ea_O2lat/R./T_points); %reaction velocity for oxygen adsorption to lattice
k_Odes=k0_Odes*exp(-Ea_O2des/R./T_points); %reaction velocity for lattice oxygen desorption
K_O2lat=k_O2lat./k_Odes; % equilibrium constant
Dp=Dp0*exp(-Ea_Dp/R./T_points);
Dv=Dv0*exp(-Ea_Dv/R./T_points);
    
%%=====================================================================================
%% Model 1: D_v*C_v ~ D_p*C_p (electron and vacancy diffusion about the same)
% Cp1_I = @(KO2s)sqrt(KO2s*Cv0*PO2I^(1/2));
% Cv1_I = @(KO2s)0.5*(CT0-Cp1_I(KO2s));
% Cp1_II = @(J,kCH4,kOdes)-kCH4/kOdes.*PCH4+1/kOdes*sqrt((kCH4.*PCH4).^2+J+Cv0.*PO2II^0.5);
% Cp1_III = @(J,kCH4,kOdes)-kCH4/kOdes.*PCH4+1/kOdes*sqrt((kCH4.*PCH4).^2+J+Cv0.*PO2II^0.5);
% Cv1_II = @(J,kCH4,kOdes) 0.5*(CT0-Cp1_II(J,kCH4,kOdes));
% 
% Cp1_Ia=zeros(1,length(T_points)); %a for array
% Cp1_IIa=zeros(1,length(T_points));
% JO2_full=zeros(1,length(T_points));
% i=1;
% %%Temperature loop 
% for(T=T_points(1):T_intervall:T_points(end))
%     T
%     JO2_full_solve=solve(J==Dp0*Dp0/2/L*(...
%         (Cv1_II(J,k_CH4(i),k_O2lat(i)))),...
%         J)
% %     JO2_full_solve=solve(J==Dp(i)*Dv(i)/2/L*(...
% %         (Cv1_II(J,k_CH4(i),k_O2lat(i))-Cv1_I(K_O2lat(i)))/(2*Dv(i)-Dp(i))+...
% %         CT0*(Dv(i)-Dp(i))/(2*Dv(i)-Dp(i))^2*...
% %         log((4*Dv(i)*Cv1_II(J,k_CH4(i),k_O2lat(i))+Dp(i)*Cp1_II(J,k_CH4(i),k_O2lat(i)))/...
% %         (4*Dv(i)*Cv1_I(K_O2lat(i))+Dp(i)*Cp1_I(K_O2lat(i))))),...
% %         J)
%     JO2_full(i)=double(real(JO2_full_solve))
%     Cp1_Ia(i)=Cp1_I(K_O2lat(i));
%     Cp1_IIa(i)=Cp1_II(JO2_full(i),k_CH4(i),k_O2lat(i));
%     Cv1_Ia(i)=Cv1_I(K_O2lat(i));
%     Cv1_IIa(i)=Cv1_II(JO2_full(i),k_CH4(i),k_O2lat(i));
%     clear JO2_full_solve;
%     i=i+1;
% end

%% Properties for Models 2, 3 & Xu/Thomson; Xu used: LSCF-6428===================
% material used: LSCF-6428
Dv0_Xu=1.58e-2;
Dv_Xu=Dv0_Xu*exp(-73.6e3/R./T_points); %[cm^2 s^-1] Data from Xu and Thomson (1999)
Dv2L=Dv_Xu/2/L; %[cm s^-1}
kf_Xu=5.90e6*exp(-226.9e3/R./T_points); %[cm atm^-0.5 s^-1]
kr_Xu=2.07e4*exp(-241.3e3/R./T_points); %[mol cm^-1 s^-1]

%% Model 2: CRR=0, D_v*C_v << D_p*C_p and C_p = C0_p (Vacancy diffusion limiting)
syms kf kr CvII;
CRR = @(kf,kr)0;

Cv2_I=zeros(length(PO2I_points),length(T_points)); %a for array
Cv2_II=zeros(length(PO2I_points),length(T_points));
JnO2_vLim=zeros(length(PO2I_points),length(T_points));
JvO2_vLim=zeros(length(PO2I_points),length(T_points));

%Temperature loop 
for(j=1:length(PO2I_points))
    PO2I=PO2I_points(j);
    PO2II=min(PO2I_points); %Take lowest value in array
    i=1;
    for(T=T_points(1):T_intervall:T_points(end))
        pre_CvII=(Dv2L(i)*kf_Xu(i)*PO2I^(0.5))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5))+kf_Xu(i)*PO2II^(0.5);
        pre_kr=2-(kf_Xu(i)*PO2I^(0.5))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5));
        Cv2II=solve(CvII*pre_CvII-...
            CRR(kf_Xu(i),kr_Xu(i))==...
            pre_kr*kr_Xu(i),...
            CvII); %[mol cm^-2]
        Cv2_II(j,i)= double(Cv2II);
        Cv2_I(j,i)=(Dv2L(i)*Cv2_II(j,i)+kr_Xu(i))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5));
        JnO2_vLim(j,i)=Dv2L(i)*(Cv2_II(j,i)-Cv2_I(j,i)); %[mol cm^-2 s^1]
        JvO2_vLim(j,i)=JnO2_vLim(j,i)*R*T0*m3ToMl*60/p0; %pV=nRT, Xu (1999) used standard conditions in order to normalize
    	i=i+1;
    end
    j=j+1;
end


%% Model using Xu & Thomson (1999) equations: D_v*C_v << D_p*C_p and C_p = C0_p (Vacancy diffusion limiting)
% material used: LSCF-6428
CvXu_I=zeros(length(PO2II_points),length(PO2I_points),length(T_points)); %a for array
CvXu_II=zeros(length(PO2II_points),length(PO2I_points),length(T_points));
JnO2_Xu=zeros(length(PO2II_points),length(PO2I_points),length(T_points));
JvO2_Xu=zeros(length(PO2II_points),length(PO2I_points),length(T_points));

for(k=1:length(PO2II_points))
    PO2II=PO2II_points(k);
    j=1;
    for(j=1:length(PO2I_points))
    PO2I=PO2I_points(j);
    i=1;
        for(T=T_points(1):T_intervall:T_points(end))
            JnO2_Xu(k,j,i)=(Dv_Xu(i)*kr_Xu(i)*(PO2I^(1/2)-PO2II^(1/2)))/...
                (2*L*kf_Xu(i)*(PO2I*PO2II)^(1/2)+Dv_Xu(i)*(PO2I^(1/2)+PO2II^(1/2)));
            JvO2_Xu(k,j,i)=JnO2_Xu(k,j,i)*R*T0*m3ToMl*60/p0; % pV=nRT, Xu (1999) used standard conditions T=273, p=1bar
            CvXu_I(k,j,i)=(JnO2_Xu(k,j,i)+kr_Xu(i))/(kf_Xu(i)*PO2I^(1/2));
            CvXu_II(k,j,i)=(JnO2_Xu(k,j,i)/Dv2L(i))+CvXu_I(k,j,i);
        	i=i+1;
        end;
    end;
end

% figure
% hold on; box on;
% for(k=1:length(PO2II_points))
%     JvO2_temp=zeros(1,length(T_points));
%     for(i=1:length(T_points))
%         JvO2_temp(i)=JvO2_Xu(k,1,i);
%     end
%     plot(T_points-273,JvO2_temp,'xk-');
% end
% xlabel('temperature, T');
% ylabel('oxygen flux, J_{O_2} [ml cm^{-2} min^{-1}]');
% hold off;
%=====================================================================================


%% Model 3: add CRR term! D_v*C_v << D_p*C_p and C_p = C0_p (Vacancy diffusion limiting)
% material by Xu LSCF-6428
syms kf kr CvII;

% CRR model
% comparable material according Wang [1995] 
kf03_CH4=10.1;  %mol/s comparable to Arai, 1986
Eaf3_CH4=97e3; % comparable to Arai, 1986
kf3_CH4=kf03_CH4*exp(-Eaf3_CH4/R./T_points);
CRR = @(kfCH4,PCH4,CvII)kfCH4*PCH4; %[mol s^-1 cm^-2]

% initialization
Cv3_I=zeros(length(PCH4_points),length(T_points)); %a for array
Cv3_II=zeros(length(PCH4_points),length(T_points));
CvII3=zeros(length(PCH4_points),length(T_points));
JnO2_3=zeros(length(PCH4_points),length(T_points));
JvO2_3=zeros(length(PCH4_points),length(T_points));
CRR_3=zeros(length(PCH4_points),length(T_points));
RHS_3=zeros(1,length(T_points));

%PCH4 loop model 3; P'=0.21, P''=0
for(j=1:length(PCH4_points))
    PO2I=PO2I_points(1);
    PO2II=PO2II_points(end);
    PCH4=PCH4_points(j);
    i=1;
    %Temperature loop model 3
    for(T=T_points(1):T_intervall:T_points(end))
        pre_CvII=(Dv2L(i)*kf_Xu(i)*PO2I^(0.5))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5))+kf_Xu(i)*PO2II^(0.5);
        pre_kr=2-(kf_Xu(i)*PO2I^(0.5))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5));
        Cv3II=solve(CvII*pre_CvII-...
            CRR(kf3_CH4(i),PCH4,CvII)==...
            pre_kr*kr_Xu(i),...
            CvII); %[mol cm^-2]
        Cv3_II(j,i)= double(Cv3II);
        Cv3_I(j,i)=(Dv2L(i)*Cv3_II(j,i)+kr_Xu(i))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5));
        JnO2_3(j,i)=Dv2L(i)*(Cv3_II(j,i)-Cv3_I(j,i)); %[mol cm^-2 s^1]
        JvO2_3(j,i)=JnO2_3(j,i)*R*T0*m3ToMl*60/p0; %pV=nRT, Xu (1999) used standard conditions in order to normalize
    	CRR_3(j,i)=CRR(kf3_CH4(i),PCH4,Cv3II);
        if(j==1)
            RHS_3(i)=pre_kr*kr_Xu(i);
        end
        CvII3(j,i)=(RHS_3(i)+CRR_3(j,i))/pre_CvII;
        i=i+1;
    end
    j=j+1;
end

%% plot parameters: Model 3
%plot reaction velocities
figure
box on;
semilogy(T_points-273,kf3_CH4,T_points-273,kf_Xu);
hold on;
semilogy(T_points-273,kr_Xu);
xlabel('temperature, T [°C]');
ylabel('reaction velocities, k [mol atm^{-x} cm^{-3} s^{-1}]');
legend('k3_{f,CH4}','k_{f,Xu}','k_{r,Xu}');
hold off;

% plot section results
%plot CRR
% figure
% hold on; box on;
% for(j=1:length(PCH4_points))
%     plot(T_points-273,CRR_3(j,:),'ok-');
% end
% plot(T_points-273,RHS_3,'xk-'); %RHS constant over j cause P' and P'' constant
% plot(T_points-273,RHS_3+CRR_3(3,:));
% xlabel('temperature, T [°C]');
% ylabel('vacancy concentration at II, C_{V(II)} [mol cm^{-3}');
% legend('CRR','RHS');
% hold off;

%plot JO2
figure
hold on; box on;
for(k=1:length(PO2II_points))
    for(i=1:length(T_points))
        JvO2_temp(i)=JvO2_Xu(k,1,i);
    end;
    plot(T_points-273,JvO2_temp,'ok--');
    for(j=1:length(PCH4_points))
        plot(T_points-273,JvO2_3(j,:),'xk-');
    end;
    legend('J_{O_2, Xu}','J_{O_2, CRR1}');
end
xlabel('temperature, T [°C]');
ylabel('oxygen flux, J_{O_2} [ml cm^{-2} min^{-1}]');
hold off;

%=====================================================================================

%% Model using Habib (2013) / Xu & Thomson (1999) equations: D_v*C_v << D_p*C_p and C_p = C0_p (Vacancy diffusion limiting)

% Dv0_Hab=1.58e-1;
% Dv_Hab=Dv0_Hab*exp(-73.6e3/R./T_points); %[cm^2 s^-1] Data from Xu and Thomson (1999)
% kf_Hab=1.11e12*exp(-226.9e3/R./T_points); %[cm atm^-0.5 s^-1]
% kr_Hab=3.85e7*exp(-241.3e3/R./T_points); %[mol cm^-1 s^-1]
% Ls_Hab=[0.8e-1 1e-1 2e-1]; %[cm]
% MWO2=32;
% A=1; %[cm^2] Habib says reactor is 1000mm long and 30 mm wide
% 
% CvHab_I=zeros(length(PO2I_points),length(T_points)); %a for array
% CvHab_II=zeros(length(PO2I_points),length(T_points));
% JnO2_Hab=zeros(length(PO2I_points),length(T_points));
% JmO2_Hab=zeros(length(PO2I_points),length(T_points));
% 
% for(k=1:length(Ls_Hab))
%     L_Hab=Ls_Hab(k);
%     j=1;
%     for(j=1:length(PO2I_points))
%         PO2I=PO2I_points(j);
%         i=1;
%         for(T=T_points(1):T_intervall:T_points(end))
%             JnO2_Hab(k,j,i)=(Dv_Hab(i)*kr_Hab(i)*(PO2I^(1/2)-PO2II^(1/2)))/...
%                 (2*L_Hab*kf_Hab(i)*(PO2I*PO2II)^(1/2)+Dv_Hab(i)*(PO2I^(1/2)+PO2II^(1/2)));
%             JmO2_Hab(k,j,i)=JnO2_Hab(k,j,i)*MWO2/1e3;
%             CvHab_I(k,j,i)=(JnO2_Hab(k,j,i)+kr_Hab(i))/(kf_Hab(i)*PO2I^(1/2));
%             CvHab_II(k,j,i)=(JnO2_Hab(k,j,i)/Dv2L(i))+CvHab_I(k,j,i);
%             i=i+1;
%         end;
%     end;
% end;

% figure
% hold on; box on;
% for(k=1:length(Ls_Hab))
%     JmO2_temp=zeros(1,length(T_points));
%     for(i=1:length(T_points))
%         JvO2_temp(i)=JmO2_Hab(k,1,i);
%     end
%     plot(T_points-273,JvO2_temp,'xk-');
% end
% xlabel('temperature, T');
% ylabel('oxygen flux, J_{O_2} [kg cm^{-2} s^{-1}]');
% hold off;
%=====================================================================================

%% Final plotting properties

% figure
% hold on; box on;
% plot(T_points-273,CvXu_I(1,:));
% plot(T_points-273,Cv2_I(1,:));
% plot(T_points-273,CvXu_II(1,:));
% plot(T_points-273,Cv2_II(1,:));
% xlabel('temperature, T');
% ylabel('C_V');
% legend('C_{V(I),Xu}','C_{V(I),2}','C_{V(II),Xu}','C_{V(II),2}');
% hold off;

%=====================================================================================

%% Final plotting results
% plot O2 flux
% figure
% hold on; box on;
% for(k=1:length(PO2II_points))
%     plot(T_points-273,JvO2_vLim(1,:),'xk-');
%     for(i=1:length(T_points))
%         JvO2_XuTemp(i)=JvO2_Xu(k,1,i);
%     end;
%     plot(T_points-273,JvO2_XuTemp,'ok--');
%     plot(T_points-273,JvO2_3(1,:),'*k-');
%     legend('JvO2_{vLim}','JvO2_{Xu}','JvO2_{3}');
% end
% xlabel('temperature, T [°C]');
% ylabel('oxygen flux, J_{O_2} [ml cm^{-2} min^{-1}]');
% hold off;