%% Time and temperature settings
%check whether new simulation or old one continues
if(t0==0)
    clearvars -except t0 t_end dt tWriteStep1 tWriteStep2 tWriteStepChange timer tic toc date...
        T0 T_low T_mid T_end nGridT=5 T_intervallLow T_intervallEnd T_points...
        orderScheme saveMode tmpSaveMode saveFilename fileNameChemicalMechanism;
else
    %Save last results to start a new simulation at final point
    finalNI_=NI_(:,:,end); finalNII_=NII_(:,:,end);
    finalPI_=PI_(:,:,end); finalPII_=PII_(:,:,end);
    oldt_points=t_points;
    oldNI_=NI_;         oldNII_=NII_;
    oldNIall_=NIall_;   oldNIIall_=NIIall_;
    oldPI_=PI_;         oldPII_=PII_;
    oldCvI_=CvI_;       oldCvII_=CvII_;
    oldJnO2_=JnO2_;     oldJnO2A_=JnO2A_;
    oldJnCH4A_=JnCH4A_;
    oldRRII_=RRII_;
    
    %delete old entries except for data that will be appended to and needed for new solution
    clearvars -except t0 t_end dt tWriteStep1 tWriteStep2 tWriteStepChange timer tic toc date...
        T0 T_low T_mid T_end nGridT=5 T_intervallLow T_intervallEnd T_points...
        orderScheme saveMode tmpSaveMode saveFilename fileNameChemicalMechanism...
        finalNI_ finalNII_ finalPI_ finalPII_...
        oldt_points oldT_points oldNI_ oldNII_ oldPI_ oldPII_ oldNIall_ oldNIIall_...
        oldCvI_ oldCvII_ oldJnO2_ oldJnO2A_ oldJnCH4A_ oldRRII_;
end
