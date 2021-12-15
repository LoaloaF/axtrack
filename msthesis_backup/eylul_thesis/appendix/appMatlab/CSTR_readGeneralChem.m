%%=========================================================================
%%  Gas phase chemistry file for 'U_permeationModelCSTRcomplexRR'
%   creates a vector containing classes of the distinct gas phase chemistry
%   class 'ReactionClass' is defined in 'reaction class
%   defines mass and computes molar fluxes
%   initializes matrices and vectors of fluxes and moles
%   determines inital partial pressures
%%=========================================================================

%% fill out positions of specie of reaction class
for(z=1:length(reacs))
    reacs(z).posReac=z;
    %go through products of single reaction
    reacs(z).posEducts=zeros(1,length(reacs(z).educts));
    if(strcmp(reacs(z).reacType,'3B'))
        reacs3B(z)=z; %saves locations of third body reactions (rrCoefs need to be modified each step)
    end
    for(k=1:length(reacs(z).educts))
        if(strcmp(reacs(z).educts(k),'M')==0) %if educt not equal to 'M'
            try
            reacs(z).posEducts(k)=find(strcmp(specieII,reacs(z).educts(k)));
            catch
                disp(strcat('ERROR! Educt=',char(reacs(z).educts(k)),' of reaction # ',...
                    num2str(z), ' appears not to be a part of the given reaction species'));
                disp('Program stopped.');
                return;
            end
        end
    end
    %go through educts of single reaction
    reacs(z).posProducts=zeros(1,length(reacs(z).products));
    for(k=1:length(reacs(z).products))
        if(strcmp(reacs(z).products(k),'M')==0)
            try
            reacs(z).posProducts(k)=find(strcmp(specieII,reacs(z).products(k)));
            catch
%                 disp(strcat('ERROR! Product=',char(reacs(z).products(k)),' of reaction # ',...
%                     num2str(z), ' appears not to be a part of the given reaction species'));
%                 disp('Species might not be part of global reaction.');
                %do not stop since product concentration not needed
            end
        end
    end
end
reacs3B(reacs3B==0) = []; %delete zero entries

%% create vector with unique keys of reactions
for(z=1:length(reacs))
    uniqueKeys_(z)={reacs(z).uniqueKey};
end

%% calculate Arrhenius for given temperatures

mixtureM=NIIall_(i,1)/(VII*qm2qcm); %calculate molar concentration of mixture (needed for TROE reaction)
for(i=1:length(T_points))
    for(z=1:length(reacs))
        rrCoeff_(z,i)=reacs(z).k(T_points(i),mixtureM);
    end
end;

%% create ReactionRateArray that calculates the reaction rates during the program
% initialize:
%allocate memory for an array of reaction rates with size of the amount of
%species in the gas phase; each entry consists of the ReactionRateClass
for(k=1:length(specieII))
    RR_(k)=ClassReactionRate; 
end;

%go through all species and establish the corresponding reaction rate for
%the whole simulation
for(k=1:length(specieII))
    RR_(k).specie=specieII(k);
    zCount=0;
    for(z=1:length(reacs))%first loop saves reachtions where species occurs
        if(ismember(char(specieII{k}), reacs(z).educts)||ismember(char(specieII{k}), reacs(z).products)) %check whether specie is a educt
            zCount=zCount+1;
            RR_(k).posReacs_(zCount)=z;
            %checks whether consumed (educt) or produced (product)
            if(ismember(char(specieII{k}), reacs(z).educts))
                RR_(k).EdOrProd_(zCount)=-1;
            else
                RR_(k).EdOrProd_(zCount)=+1;
            end
        end
    end
    %save number of equations where species appears
    RR_(k).nReacs=length(RR_(k).posReacs_);
    %loop through the reactions where the species k occurs
    for(z=1:RR_(k).nReacs)
        zz=RR_(k).posReacs_(z); %zz is number of containing reaction
        if( RR_(k).EdOrProd_(z)==-1)
            RR_(k).nOccur_(z)=[length(find(strcmp(reacs(zz).educts,specieII(k))))];
            for(zzz=1:length(reacs(zz).educts)) %zzz is iterating number of species
                if(strcmp(specieII(k),reacs(zz).educts(zzz)))
                    RR_(k).stoiSpecie_(z)=reacs(zz).stoiEducts(zzz);
                end
            end
        end
        if( RR_(k).EdOrProd_(z)==+1)%if specie is a product of reaction z resp. zz
            RR_(k).nOccur_(z)=[length(find(strcmp(reacs(zz).products,specieII(k))))];
            for(zzz=1:length(reacs(zz).products))
                if(strcmp(specieII(k),reacs(zz).products(zzz)))
                    RR_(k).stoiSpecie_(z)=reacs(zz).stoiProducts(zzz);
                end
            end
        end
    end
end


%% plot reaction rate from Arrhenius
% fig=figure;
% for(i=1:length(T_points))
%     for(z=1:length(reacs))
%         kk(z,i)=reacs(z).k(T_points(i));
%     end
% end
% box on;
% semilogy(1./(T_points-T0),kk(1,:));
% if(length(reacs(1).educts)==2);
%     legendEntries(1)=strcat(reacs(1).educts(1),',',reacs(1).educts(2));
% else
%     legendEntries(1)=strcat(reacs(1).educts(1));
% end;
% 
% hold on
% for(z=2:length(reacs))
%     semilogy(1./(T_points-T0),kk(z,:));
% if(length(reacs(z).educts)==2);
%     legendEntries(z)=strcat(reacs(z).educts(1),',',reacs(z).educts(2));
% else
%     legendEntries(z)=strcat(reacs(z).educts(1));
% end;
% end
% xlabel('inverse temperature, 1/T [K^{-1}]');
% ylabel('reaction velocity, k [mol cm^{-3} s^{-1}]')
% legend(legendEntries);
% hold off
% clear kk;
