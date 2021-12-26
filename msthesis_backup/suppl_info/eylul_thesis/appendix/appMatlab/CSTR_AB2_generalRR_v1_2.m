%%=========================================================================
%%  Research of New CvII Model using stirred reactor
%   This script uses different variables to check the following oxygen
%   permeation model through a membrane
%   - D_v*C_v << D_p*C_p and C_p = C0_p (Vacancy diffusion limiting)
%   - 1: CvII*pre_CvII - CRR = pre_kr*kr
%   - 2: CvI=(D_v/2/L*C_v(II)+k_r)/(D_v/2/L+k_f*P_O2^0.5)
%   - 3: J_O2 = D_v/2/L (C_v(II)-C_v(I))
%   
%   On both sides of the membrane is a constant flow of air (feed) and
%   reducing agent (sweep)
%   - Reactor operates with constant temperature and constant pressure
%   - Partial pressures are determined by the continously stirred reactor
%
%   A 21 reactions mechanism for the oxidation of methane is used
%
%   ANNOTATION:
%   Reprogrammed version
%   This version uses Euler method for time integration
%   The new version contains a mmethod to save distinct iteration steps
%%=========================================================================

%% CRR model
% comparable material according Wang [1995] 
kf0_CH4=10.1;  %[mol/s/cm^3]; multiplication by thickness
Eaf_CH4=97e3; % comparable to Arai, 1986
kf_CH4=kf0_CH4*exp(-Eaf_CH4/R./T_points);
CRR = @(kfCH4,PCH4)kfCH4*PCH4*CRRthickness; %[mol s^-1 cm^-2 um^-1]; L is in cm * times "volume factor"

%% ========================================================================
%  Formulate time dependent equations for CSTR
syms F_inSym F_outSym RRsym 
dNdt = @(F_inSym,F_outSym,RRsym) F_inSym - F_outSym + RRsym;

%  Time-dependent calculation of moles per volume
disp(['The program will run for ',num2str(length(T_points)),' temperature and ',num2str(ntPointsFull),' time points.']);
disp('Start temperature and time loop...');
for(i=1:length(T_points))
    toc
    disp(['reactor calculation for temperature number ',num2str(i),' out of ',num2str(length(T_points)),' at ',num2str(T_points(i)),' K.'])
    writeI=1; writeII=1; z=1; %this is for recording purpose
    
    %set initial values for distinct temperature===========================
    %feed side----------------------JnCH4A_---------------------------------------
    for(k=1:length(specieI))
            tmpPI_(k,1)=PI_(k,i,1);
            tmpNI_(k,1)=NI_(k,i,1);
    end
    %sweep side------------------------------------------------------------
    for(k=1:length(specieII))
            tmpPII_(k,1)=PII_(k,i,1);
            tmpNII_(k,1)=NII_(k,i,1);
    end
    %calculate overall moles
            tmpNIall_(1)=sum(tmpNI_(:,1)); %calculate total amount of moles in volume I
            tmpNIIall_(1)=sum(tmpNII_(:,1)); %calculate all moles in volume II
    
    %start loop for time discretization====================================
    jjj=1; %jjj is the time step that counts the reduced steps determined by variable "tWriteStep1"
    t=t0;
    for(j=1:ntPointsFull-1)
%         dt=t_pointsFull(j+1)-t_pointsFull(j); %use for varying time step
        t=t+dt;
        %if scheme is of higher order, a "jj-loop" needs to be added
        for(jj=1:orderScheme); %this is a temporary time step and can be changed in a loop if more order scheme is used
            %calculate oxygen flux
            %if PO2II becomes negative there preCvII becomes imaginary number!
            %jj needs to be adjusted to  jj=1:1:orderScheme
            PO2I=tmpPI_(1,jj)/p0atm;    %ATTENTION on value storage of different partial pressures!
            PO2II=tmpPII_(1,jj)/p0atm;
            PCH4=tmpPII_(3,jj)/p0atm;
            PCH3=tmpPII_(4,jj)/p0atm;
            pre_CvII=(Dv2L(i)*kf_Xu(i)*PO2I^(0.5))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5))+kf_Xu(i)*PO2II^(0.5); %pressures in atm!
            pre_kr=2-(kf_Xu(i)*PO2I^(0.5))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5));
            tmpCvII_(jj)=(CRR(kf_CH4(i),PCH4)+pre_kr*kr_Xu(i))/pre_CvII;
            tmpCvI_(jj)=(Dv2L(i)*tmpCvII_(jj)+kr_Xu(i))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5));
            tmpJnO2_(jj)=Dv2L(i)*(tmpCvII_(jj)-tmpCvI_(jj)); %[mol cm^-2 s^-1]
            tmpJnO2A_(jj)=tmpJnO2_(jj)*A*m2ToCm2; %[mol/s]
            
            % calculate fluxes to and from mambrane
            tmpJnCH4A_(jj)=kf_CH4(i)*PCH4*A*m2ToCm2*CRRthickness;
            tmpJnO2Ared_(jj)=tmpJnO2A_(jj)-0.25*tmpJnCH4A_(jj); %oxygen flux
            JnCH3A=tmpJnCH4A_(jj);
            JnH2OA=0.5*tmpJnCH4A_(jj);
            
            %==================================================================
            %calculate feed side
            MWmixI=0;
            for(k=1:length(specieI))
                MWmixI = MWmixI+tmpPI_(k,jj)/p0atm*MW(char(specieI(k)));
            end
            massFluxIOutlet=massFluxIin-tmpJnO2A_(jj)*MW('O2');
            molarFluxIOutlet = massFluxIOutlet/MWmixI;
            for(k=1:length(specieI))
                if(strcmp(char(specieI{k}),'O2'))
                    F_in  = molarFluxIin*PO2I0;
                    F_out = molarFluxIOutlet*tmpPI_(k,jj)+tmpJnO2A_(jj);
                    RR=0;
%                 if(F_in<JnO2A(i,j)) display('Membrane permeation is higher than oxygen supported!'); end;
                elseif(strcmp(char(specieI{k}),'N2'))
                    F_in  = molarFluxIin*PN2I0;
                    F_out = molarFluxIOutlet*tmpPI_(k,jj);
                    RR=0;
                end
            
                if(jj==1)
                    tmpNI_(k,jj+1)=tmpNI_(k,jj)+dt*dNdt(F_in,F_out,RR);
                    tmpdNdtI_(k)=dNdt(F_in,F_out,RR);
                else
                    tmpNI_(k,jj+1)=tmpNI_(k,1)+dt/2*(dNdt(F_in,F_out,RR)+tmpdNdtI_(k));
                end
                %check that values are not negative at (I)!
                if(tmpNI_(k,jj+1)<0)
                    if(writeI==1)
                        dispStr=strcat({'At time t='},num2str(t),{' sec, the total moles in tank volume of specie '},char(specieI{k}),{' at temperature '},num2str(T_points(i)),...
                        {' K is negative at sweep side (I) with value '},num2str(tmpNI_(k,jj+1)),'!');
                        dispArray{z}=dispStr;
                        clear dispStr;
                        writeI=0;
                        z=z+1;
                    end
                    tmpNI_(k,jj+1)=0;
                end
            end
        
            %==================================================================
            %calculate sweep side
            % calculate the molar weight of the gas mixture
            
            MWmixII=0;
            for(k=1:length(specieII))
            MWmixII = MWmixII+tmpPII_(k,jj)/p0atm*MW(char(specieII(k)));
            end
        
            %calculate overall fluxes
            massFluxIIinAll = massFluxIIin+tmpJnO2Ared_(jj)*MW('O2')+JnCH3A*MW('CH3')+JnH2OA*MW('H2O')-tmpJnCH4A_(jj)*MW('CH4'); %[g/s]
            molarFluxIIOutlet = massFluxIIinAll/MWmixII;
            
            %calculate single reaction rates at time step for each
            %elemantary reaction of mechanism
            singleRR_=zeros(1,length(reacs));
            for(z=1:length(reacs))
                singleRR_(z)=reacs(z).calcSingleRR(tmpPII_(:,jj),specieII,rrCoeff_);
            end;
            tmpSRR_(:,jj)=singleRR_;
            
            %calculate single species
            for(k=1:length(specieII))
                %----------------------------------------------------------    
                %special treatment for species connected to catalytic
                %reaction at membrane surface and inlet
                tmpRR      = RR_(k).calculateRR(singleRR_); %no nesting allowed by Matlab
                RR         = tmpRR*tmpNIIall_(jj);
                if(strcmp(char(specieII{k}),'O2'))
                    F_in   = molarFluxIIin*PO2II0/p0atm+tmpJnO2Ared_(jj);
                    F_out  = molarFluxIIOutlet*tmpPII_(k,jj);
                    if(strcmp(fileNameChemicalMechanism,'ChemMechanism_2Reac_modA.m'))
                        RR = 100*tmpRR*tmpNIIall_(jj);
                    end
                elseif(strcmp(char(specieII{k}),'N2'))
                    F_in   = molarFluxIIin*PN2II0/p0atm;
                    F_out  = molarFluxIIOutlet*tmpPII_(k,jj);
                    RR     = 0; %only exception since intert!
                elseif(strcmp(char(specieII{k}),'CH4'))
                    F_in   = molarFluxIIin*PCH40/p0atm;
                    F_out  = molarFluxIIOutlet*tmpPII_(k,jj)+tmpJnCH4A_(jj);
                elseif(strcmp(char(specieII{k}),'CH3'))
                    F_in   = JnCH3A;
                    F_out  = molarFluxIIOutlet*tmpPII_(k,jj);
                elseif(strcmp(char(specieII{k}),'H2O'))
                    F_in   = JnH2OA;
                    F_out  = molarFluxIIOutlet*tmpPII_(k,jj);
                else
                    F_in   = 0;
                    F_out  = molarFluxIIOutlet*tmpPII_(k,jj);
                end
                %----------------------------------------------------------
                tmpRRII_(k,jj)=RR;
                if(jj==1)
                    tmpNII_(k,jj+1)=tmpNII_(k,jj)+dt*dNdt(F_in,F_out,RR);
                    tmpdNdtII_(k)=dNdt(F_in,F_out,RR);
                else
                    tmpNII_(k,jj+1)=tmpNII_(k,1)+dt/2*(dNdt(F_in,F_out,RR)+tmpdNdtII_(k));
                end
                if(tmpNII_(k,jj+1)<0)
                    if(writeII==1)
                        dispStr=strcat({'At time t='},num2str(t),{' sec, the total moles in tank volume of specie '},char(specieII{k}),{' at temperature '},num2str(T_points(i)),...
                        {' K is negative at sweep side (II) with value '},num2str(tmpNII_(k,jj+1)),'!');
                        dispArray{z}=dispStr;
                        clear dispStr;
                        writeII=0;
                        z=z+1;
                    end
                    tmpNII_(k,jj+1)=0;
                end
            end
            %==============================================================
            %calculate intermediate partial pressures
            %calculate overall amount of moles
            tmpNIall_(jj+1)=sum(tmpNI_(:,jj+1)); %calculate total amount of moles in volume I
            tmpNIIall_(jj+1)=sum(tmpNII_(:,jj+1)); %calculate all moles in volume II
            %partial pressures on feed side
            for(k=1:length(specieI))
                tmpPI_(k,jj+1)=tmpNI_(k,jj+1)/tmpNIall_(jj+1); %[atm] calcualted from moles in reactor for p=1
            end
            %partial pressures on sweep side
            for(k=1:length(specieII))
                tmpPII_(k,jj+1)=tmpNII_(k,jj+1)/tmpNIIall_(jj+1); %[atm] calcualted from moles in reactor for p=1
                if(tmpPII_(k,jj+1)<0) %actually not necessary since correction term is done above
                    tmpPII_(k,jj+1)=0;
                end
            end
            %==============================================================
        end
        %------------------------END SCHEME LOOP---------------------------

        
        %===============MASS CORRECTION====================================
            %introduce a correction term (assure constant mass), feed
        if(jj==orderScheme && rem(j,10)==0) % do correction every 10 time steps
            totalMassI_tdt=0;
            for(k=1:length(specieI))
                totalMassI_tdt=totalMassI_tdt+tmpNI_(k,jj+1)*MW(char(specieI(k)));
            end
            for(k=1:length(specieI))
                tmpNI_(k,jj+1)=tmpNI_(k,jj+1)*(m0I(i)/totalMassI_tdt);
            end
            %introduce correction term, sweep
            totalMassII_tdt=0;
            for(k=1:length(specieII))
                    totalMassII_tdt=totalMassII_tdt+tmpNII_(k,jj+1)*MW(char(specieII(k)));
            end
            
            % the correction factor epsilon is epsilonII=m0II(i)/totalMassII_tdt;
            for(k=1:length(specieII))
                    tmpNII_(k,jj+1)=tmpNII_(k,jj+1)*(m0II(i)/totalMassII_tdt);
            end
        end
        %==================================================================
            %calculate partial pressures for t+dt
            %calculate
            tmpNIall_(jj+1)=sum(tmpNI_(:,jj+1)); %calculate total amount of moles in volume I
            tmpNIIall_(jj+1)=sum(tmpNII_(:,jj+1)); %calculate all moles in volume II
        
            %calculate partial pressures for t+dt
            %partial pressures on feed side
            for(k=1:length(specieI))
                tmpPI_(k,jj+1)=tmpNI_(k,jj+1)/tmpNIall_(jj+1); %[atm] calcualted from moles in reactor for p=1
            end
            %partial pressures on sweep side
            for(k=1:length(specieII))
                tmpPII_(k,jj+1)=tmpNII_(k,jj+1)/tmpNIIall_(jj+1); %[atm] calcualted from moles in reactor for p=1
                if(tmpPII_(k,jj+1)<0) %actually not necessary since correction term is done above
                    tmpPII_(k,jj+1)=0;
                end
            end
        %% =================WRITE AND MODIFY DATA==========================
            if(jj==orderScheme)
                %----------------------------------------------------------
                %transfer data one back for new iteration
                tmpNIall_(1)=tmpNIall_(end);
                tmpNIIall_(1)=tmpNIIall_(end);
                for(k=1:length(specieI))
                    tmpNI_(k,1)=tmpNI_(k,end);
                    tmpPI_(k,1)=tmpPI_(k,end); 
                end
                for(k=1:length(specieII))
                    tmpNII_(k,1)=tmpNII_(k,end);
                    tmpPII_(k,1)=tmpPII_(k,end);
                    tmpRRII_(k,1)=tmpRRII_(k,end);
                end
                %==========================================================
                %Write data to matrices
                if((rem(j,tWriteStep1)==0 && j*dt<=tWriteStepChange) || (rem(j,tWriteStep2)==0 && j*dt>=tWriteStepChange)) %checks whether the remainder after division be tWriteStep1 equals 0
                    if(j*dt<=tWriteStepChange) %increase write step for higher values of t (improves log-plots)
                        tWriteStep=tWriteStep1;
                    else
                        tWriteStep=tWriteStep2;
                    end
                    jjj=jjj+1;
                    CvI_(i,jjj)=tmpCvI_(1); %only take first value
                    CvII_(i,jjj)=tmpCvI_(1);
                    JnO2_(i,jjj)=tmpJnO2_(1);
                    JnO2A_(i,jjj)=tmpJnO2A_(1);
                    JnCH4A_(i,jjj)=tmpJnCH4A_(1);
                    NIall_(i,jjj)=tmpNIall_(end);
                    NIIall_(i,jjj)=tmpNIIall_(end);
                    NI_(:,i,jjj)=tmpNI_(:,end);
                    PI_(:,i,jjj)=tmpPI_(:,end);
                    NII_(:,i,jjj)=tmpNII_(:,end);
                    PII_(:,i,jjj)=tmpPII_(:,end);
                    RRII_(:,i,jjj)=tmpRRII_(:,2);
                    sRRII_(:,i,jjj)=tmpSRR_(:,1);
                    %save time-------------------
                    t_points(jjj)=t;
                    
                    %save temporary results in 'saveFilename'
                    %execute only every 100 points that are saved
                    if(strcmp(tmpSaveMode,'on') && rem(jjj,100)==0)
                        try
                            save(saveFilename);
                        catch
                            disp('Data could not me saved temporarily. Check filename');
                        end
                    end
                end
                
                %----------------------------------------------------------
                %saves last point after data has transferred to final point
                if(j==ntPointsFull-1) %saves last point
                    PO2I=tmpPI_(1,1)/p0atm;    %ATTENTION on value storage of different partial pressures!
                    PO2II=tmpPII_(1,1)/p0atm;
                    PCH4=tmpPII_(3,1)/p0atm;
                    PCH3=tmpPII_(4,1)/p0atm;
                    pre_CvII=(Dv2L(i)*kf_Xu(i)*PO2I^(0.5))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5))+kf_Xu(i)*PO2II^(0.5); %pressures in atm!
                    pre_kr=2-(kf_Xu(i)*PO2I^(0.5))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5));
                    CvII_(i,length(t_points))=(CRR(kf_CH4(i),PCH4)+pre_kr*kr_Xu(i))/pre_CvII;
                    CvI_(i,length(t_points))=(Dv2L(i)*CvII_(i,length(t_points))+kr_Xu(i))/(Dv2L(i)+kf_Xu(i)*PO2I^(0.5));
                    JnO2_(i,length(t_points))=Dv2L(i)*(CvII_(i,length(t_points))-CvI_(i,length(t_points))); %[mol cm^-2 s^-1]
                    JnO2A_(i,length(t_points))=JnO2_(i,length(t_points))*A*m2ToCm2; %[mol/s]
                    JnCH4A_(i,length(t_points))=kf_CH4(i)*PCH4*A*m2ToCm2*CRRthickness;
                    NIall_(i,length(t_points))=tmpNIall_(1);
                    NIIall_(i,length(t_points))=tmpNIIall_(1);
                    for(k=1:length(specieI))
                        NI_(k,i,length(t_points))=tmpNI_(k,1);
                        PI_(k,i,length(t_points))=tmpPI_(k,1);
                    end
                    for(k=1:length(specieII))
                        NII_(k,i,length(t_points))=tmpNII_(k,1);
                        PII_(k,i,length(t_points))=tmpPII_(k,1);
                        RRII_(k,i,length(t_points))=tmpRRII_(k,1);
                    end
                    %write time vector
                    t_points(end)=t_end;
                end
                    
            end
            %==============================================================
            
            
            if(j==fix(0.05*ntPointsFull))
                toc
                disp('5% of time steps calculated.')
            elseif(j==fix(0.1*ntPointsFull))
                toc
                disp('10% of time steps calculated.')
            elseif(j==fix(0.25*ntPointsFull))
                toc
                disp('25% of time steps calculated.')
            elseif(j==fix(0.5*ntPointsFull))
                toc
                disp('50% of time steps calculated.')
            elseif(j==fix(0.75*ntPointsFull))
                toc
                disp('75% of time steps calculated.')
            end
    end
                
    %----------------------------END TIME LOOP-----------------------------
    if(exist('dispArray'));
        for(z=1:length(dispArray));
            disp(char(dispArray{z}));
        end;
    end;
end
%-----------------------------END TEMP LOOP--------------------------------
%% merge previous and new solution and save
if(t0~=0)
    try
        oldt_points(end)=[];
        t_points=cat(2,oldt_points,t_points);
        oldNI_(:,:,end)=[];
        NI_=cat(3,oldNI_,NI_);
        oldNII_(:,:,end)=[];
        NII_=cat(3,oldNII_,NII_);
        oldPI_(:,:,end)=[];
        PI_=cat(3,oldPI_,PI_);
        oldPII_(:,:,end)=[];
        PII_=cat(3,oldPII_,PII_);
        JnO2_(:,1)=[]; %SAVE FIRST OF NEW ONE, SINCE FLUX NOT SAVED AS INITIAL VALUE
        JnO2_=cat(2,oldJnO2_,JnO2_);
        JnO2A_(:,1)=[];
        JnO2A_=cat(2,oldJnO2A_,JnO2A_);
        JnCH4A_(:,1)=[];
        JnCH4A_=cat(2,oldJnCH4A_,JnCH4A_);
        RRII_(:,:,1)=[];
        RRII_=cat(3,oldRRII_,RRII_);
    catch
        disp('An error occured while merging old and new data. Some data might not have been saved.');
    end
end

%% delete not needed properties
clear tmpCvI_ tmpCvI_ tmpJnO2_ tmpJnO2A_ tmpJnCH4A_ tmpNIall_ tmpNIIall_ tmpNI_ tmpPI_ tmpNII_ tmpPII_ tmpRRII_;
clear tmp_reac  totalMassI_tdt totalMassII_tdt tRefine write i j jj z;
clear F_in F_in_ F_out F_out_ fig;
clear massFluxIIinAll massFluxIOutlet molarFluxIIin molarFluxIin molarFluxIIOutlet molarFluxIOutlet;
clear PO2I PO2II PCH4 PCH3;
clear pre_CvII pre_kr;
clear MWmixI MWmixII

% clear ans atmToBar barToPa calToJ kcalTokJ m2ToCm2 m3ToCm3 m3ToMl dNdt;
% clear Dv0_Xu k L NA;
% clear dt_1st dt T_intervallEnd T_intervallLow t_end T_end T_low T_mid t_points1st nGridT VI VII;


%% end program
% disp(strcat({'Calculation with end time '},num2str(t),{' and time step dt='},num2str(dt),{' ended.'}));
