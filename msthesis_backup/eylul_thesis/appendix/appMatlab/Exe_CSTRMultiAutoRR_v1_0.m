%%=========================================================================
%%  Research of New CvII Model using stirred reactor
%   This script is the execution file of several different times in a row
%%=========================================================================

format long;
tic
disp(datestr(clock));
timer{1}=datestr(datetime('now'));
date=char(datetime('now','TimeZone','local','Format','yyMMdd'));
disp('Start program "CSTR_membrane_permeation_CRR_2.2" and initialize start values');

saveMode = 'on'; %save mode saves workspace; can be 'on' of 'off'
tmpSaveMode = 'on'; %save data in between with save Filename

if(strcmp(saveMode,'on') && strcmp(tmpSaveMode,'on'))
    disp('Data will be saved during the simulation and at the end of the program.')
elseif(strcmp(saveMode,'on') && strcmp(tmpSaveMode,'on')==0)
    disp('Data will only be saved once the program has finished.')
end
% Load data from where to proceed------------------------------------------
% load('161008_AB2_gRR_Sun08_t6e-02_dt1e-08_T873.15_VA4PLQI10_01PCH4_Wang95.mat');

%% Set Temperatures--------------------------------------------------------
T_low=650; %[C]
T_end=1200; %[C]
% T_points=[650 750 850 950 1000 1050 1100 1150 1200]; T_points=T_points+273.15;
T_points=[1000]+273.15; %[K]

%% Set chemical mechanism--------------------------------------------------
fileNameChemicalMechanism='ChemMechanism_PetRogg93.m';

%% Set time, time step and the write step----------------------------------
% first calculation
% t0=0;
t0=t_points(end);
t_end=1.1e-3;           %[s]
dt=1e-8;               %time step; Note: solution might become imaginary if dt too small!
tWriteStep1=10;        %this step defines that the data of each tWriteStep^th time step plus the first and last one will be saved
tWriteStep2=1000;       %less refined write step for higher times
tWriteStepChange=1.0e-3;  %time at which writing changes from tWriteStep1 to tWriteStep2
orderScheme=2;          %determines order of scheme and the size of temporary variables (always save j until j+order Scheme steps)
saveFilename = char(strcat(cellstr(date),'_tmp_AB2_gRR_PR93_MC_t',num2str(t_end,'%10.0e\n'),'_dt',...
    num2str(dt,'%10.0e\n'),'_T',num2str(T_points(1)),'.mat'));
disp('------------------------------------------------------------------');
disp(strcat('First simulation with start time t0=',num2str(t0),', end time t_end=',...
    num2str(t_end),' and time step dt=',num2str(dt)));

run('CSTR_setOldSolution.m');
run(fileNameChemicalMechanism); %implements chemical mechanism
run('CSTR_settings.m');
run('CSTR_readGeneralChem.m');
run('CSTR_AB2_generalRR_v1_2.m');


% %% last calculation----------------------------------------------------------
% t0=t_end;
% t_end=1.00e-1;           %[s]
% dt=5e-7;               %time step; Note: solution might become imaginary if dt too small!
% tWriteStep=100;       %this step defines that the data of each tWriteStep^th time step plus the first and last one will be saved
% orderScheme=2;          %determines order of scheme and the size of temporary variables (always save j until j+order Scheme steps)
% disp('------------------------------------------------------------------');
% disp(strcat('Final simulation with start time t0=',num2str(t0),', end time t_end=',...
%     num2str(t_end),' and time step dt=',num2str(dt)));
% run('CSTR_settings.m');
% run('CSTR_readGeneralChem.m');
% run('CSTR_AB2_generalRR_v1_2.m')

%% delete data not needed
clear oldNI_ oldNII_ oldPI_ oldPII_ oldt_points finalNI_ finalNII_ finalPI_ finalPII_ oldNIall_ oldNIIall_...
        oldCvI_ oldCvII_ oldJnO2_ oldJnO2A_ oldJnCH4A_ oldRRII_;
clear RRHO2 RRHCO_4 RRHCO_3 RRHCO_2 RRHCO_1 RRH2 RRH RRCO_3 RRCO_2 RRCO_1 RRCH4_3 RRCH4_2n ...
    RRCH4_1 RRCH3_2 RRCH3_1 RRCH2O_3 RRCH2O_2 RRCH2O_1 RR2OH RR RRO RROH;

saveFilename = char(strcat(cellstr(date),'_AB2_gRR_PR93_MC_t',num2str(t_points(end),'%10.0e\n'),'_dt',...
    num2str(dt,'%10.0e\n'),'_T',num2str(T_points(1)),'_VA4PLQI10_01PCH4_Wang95.mat'));

if(strcmp(saveMode,'on'))
    save(saveFilename);
end

%% ------------------------------------------------------------------------
disp('Program ended.');
toc
timer{2}=datestr(clock);
timer{3}=toc;
