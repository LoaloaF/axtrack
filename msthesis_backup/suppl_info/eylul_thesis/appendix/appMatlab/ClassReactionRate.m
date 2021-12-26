classdef ClassReactionRate < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        specie;
        posReacs_;
        EdOrProd_; %-1 indicated species belongs to educts, +1 to products
        stoiSpecie_; %saves the stoichiometric coefficients of specie in each participating reaction
		nReacs;
        nOccur_; %counts number of occurance of specie in single reaction;
                 %stoichiometric coefficient could be 1 but species shows
                 %up twice; Note: if occurence multiple stoich MUST be 1!
        
    end
    
    methods
        function obj = ReactionClass()
            % all initializations, calls to base class, etc. here,
        end
        
        function RR = calculateRR(obj,singleRR_) %gives class properties and temperature as input
            RR=0;
            for(z=1:obj.nReacs)
                RR=RR+singleRR_(obj.posReacs_(z))*obj.stoiSpecie_(z)*obj.nOccur_(z)*obj.EdOrProd_(z);
            end
        end
        
    end
    
end

