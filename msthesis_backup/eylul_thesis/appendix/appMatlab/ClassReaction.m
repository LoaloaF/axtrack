classdef ClassReaction < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        posReac;
        educts;
        stoiEducts;
        posEducts;      %stores position of single educts of specie array
        products;
        stoiProducts;
        posProducts
        reacType;
        thirdBodyMap;
        thirdBodyEff;
        uniqueKey;
        k0;
        b;
        Ea;
    end
    
    methods
        function obj = ReactionClass()
            % all initializations, calls to base class, etc. here,
        end
        
        %create method that calculates the reaction rate velocity
        %M [mol/qcm] is the molar concentration of the mixture
        function kValue = k(obj,T,M) %gives class properties and temperature as input
            %calculate for Lindemann-Hinshelwood or TROE reaction velocity
            if(strcmp(obj.reacType,'TROE') || strcmp(obj.reacType,'TROELin'))
                k_0=obj.k0(1)*T^obj.b(1)*exp(-obj.Ea(1)/8.3147/T);
                k_inf=obj.k0(2)*T^obj.b(2)*exp(-obj.Ea(2)/8.3147/T);
                Pr=k_0*M/k_inf; %ASSUMPTION: M constant all the time
                    if(strcmp(obj.reacType,'TROE'))
                    %at the moment rather hard coded
                    p0c=1.01325e5; %assume constant pressure!
                    Fcent=0.577*exp(-T/2370.0);
                    Nhat=0.75-1.27*log10(Fcent);
                    logF=log10(Fcent)/(1+(log10(k_0*(p0c/(8.3147*T))/k_inf)/Nhat)^2);
                    F=10^logF;
                else
                    F=1;
                end
                kValue = k_inf*(Pr/(1+Pr))*F;
            else
                kValue = obj.k0*T^obj.b*exp(-obj.Ea/8.3147/T);
            end
        end
        %% ==========Calculate a Single Reaction Rate======================
        function singleRR = calcSingleRR(obj,X_,specie_,k_)
            singleRR=1;
            %--------------------------------------------------------------
%             if(ismember('M', obj.educts))
              if(strcmp(obj.reacType,'3B')) %faster than other if clause
                XM=0;
                for(k=1:length(specie_))
                    %XM is the thirds body collision efficiency sum(a[X])
                    if(ismember(specie_(k),keys(obj.thirdBodyMap)))
                    	XM=XM+X_(k)*obj.thirdBodyMap(char(specie_(k)));
                    else
                        XM=XM+X_(k)*obj.thirdBodyMap('other');
                    end
                end
                obj.thirdBodyEff=XM;
                singleRR=XM; %fill XM, third body collision efficiency
                for(k=1:length(obj.educts)) %k is species
                    if(strcmp(obj.educts(k),'M')==0) %explude M
                        singleRR=singleRR*X_(obj.posEducts(k))^(obj.stoiEducts(k));
                    end
                end
                %(+-) ny_specie*prod(P^ny)*k(T)
                singleRR=singleRR*k_(obj.posReac);
            %--------------------------------------------------------------
            else %if no third body reaction is involved
                for(k=1:length(obj.educts)) %k is species
                    singleRR=singleRR*X_(obj.posEducts(k))^(obj.stoiEducts(k));
                end
                %finally multiply by reaction velocity k=A*T^b*exp(-Ea/R/T)
                singleRR=singleRR*k_(obj.posReac);
            end
            
        end
            
        
    end
    
end

