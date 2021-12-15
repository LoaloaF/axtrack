/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2.4.0                                 |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvOptions;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

multiSource
{
    type            scalarCodedSource;

    active          true;
    selectionMode   all;
    //cellZone		injZone;

    scalarCodedSourceCoeffs
    {
	selectionMode   all;
        fieldNames      (rho O2 CH4 CH3 H2O h); //fieldI cases
        redirectType    sourceTime;
		
        codeInclude
        #{
		#include "OFstream.H"
		#include "ListOps.H"

		//[J/g/K]
		double cp_O2  (double T){double cp; cp= 3.98840075429294e-11*pow(T,3.0)-2.26878952996114e-7*pow(T,2.0)+4.92790571515887e-4*T+0.781908472710344;return cp;}
		double cp_H2O (double T){double cp; cp=-7.94999046797271e-11*pow(T,3.0)+2.22487452357969e-7*pow(T,2.0)+4.51360920433265e-4*T+1.691816202097;return cp;}
		double cp_CH3 (double T){double cp; cp= 7.47665972334058e-11*pow(T,3.0)-8.13669015616975e-7*pow(T,2.0)+2.89082358663306e-3*T+1.76820147961851;return cp;}
		double cp_CH4 (double T){double cp; cp= 1.63165737618806e-10*pow(T,3.0)-1.52695287005592e-6*pow(T,2.0)+5.19514956742978e-3*T+0.741966170624295;return cp;}
		//--------------------CONSTANTS------------------------------//
		const double L = 0.1; //(cm)
		const double R = 8.31446;
		//--------------------FUNCTIONS------------------------------//
		double D_v    (double T){double Dv; const double Dv0 = 4.98e-3; const double Ea = 5.96e4; Dv = Dv0*exp(-Ea/(R*T)); return Dv;} //Hunt
		double k_f    (double T){double kf; const double kf0 = 1.013; const double Ea = 7.810e4; kf = kf0*exp(-Ea/(R*T)); return kf;}
		double k_r    (double T){double kr; const double kr0 = 2.010e-3; const double Ea = 1.106e5; kr = kr0*exp(-Ea/(R*T)); return kr;}
		double k_CH4  (double T){double kCH4; const double kCH40 = 10.1; const double Ea = 97e3; kCH4 = kCH40*exp(-Ea/(R*T)); return kCH4;}
		double pre_CvII (double T, double Dv, double kf, double PO2feed, double PO2sweep){
				 double pCvII; pCvII = ((Dv/(2*L)*kf*pow(PO2feed,0.5))/(Dv/(2*L)+kf*pow(PO2feed,0.5)))+kf*pow(PO2sweep,0.5); return pCvII;}
		double pre_kr	(double T, double Dv, double kf, double PO2feed){
				 double pkr; pkr = 2-(kf*pow(PO2feed,0.5))/(Dv/(2*L)+kf*pow(PO2feed,0.5)); return pkr;}
		double CRR	(double kCH4, double PCH4){
				double CRR; CRR = kCH4*PCH4; return CRR;}
        #};

        codeCorrect
        #{
        #};
		codeAddSup
        #{
        	Pout<< "**codeAddSup (rho O2 CH4 CH3 H2O h)**" << endl;

        	//=====================CELL MANAGEMENT==========================//
			//get cell zone cells, sweep
			const vectorField& C = mesh_.C();
			const label cellZoneIDInj1 = mesh().cellZones().findZoneID("injZoneI");
			const labelList& cellListInj1 = mesh().cellZones()[cellZoneIDInj1];
			const label cellZoneIDInj2 = mesh().cellZones().findZoneID("injZoneII");
			const labelList& cellListInj2 = mesh().cellZones()[cellZoneIDInj2];
			labelList cellListInj = cellListInj1;
			cellListInj.append(cellListInj2);
			//feed
			const label cellZoneIDSink1 = mesh().cellZones().findZoneID("sinkZoneI");
			const labelList& cellListSink1 = mesh().cellZones()[cellZoneIDSink1];
			const label cellZoneIDSink2 = mesh().cellZones().findZoneID("sinkZoneII");
			const labelList& cellListSink2 = mesh().cellZones()[cellZoneIDSink2];
			labelList cellListSink = cellListSink1;
			cellListSink.append(cellListSink2);
			//----------------
			const volScalarField& T = mesh().lookupObject<volScalarField>("T");
			const volScalarField& pO2 = mesh().lookupObject<volScalarField>("pO2");
			const volScalarField& pCH4 = mesh().lookupObject<volScalarField>("pCH4");
			//get all cells
			scalarField& sourceI = eqn.source();
			//get fv surface at membrane
			const label cellPatchIDBaffle = mesh().boundaryMesh().findPatchID("baffle");
			const scalarField& surfA_Baffle = mesh().magSf().boundaryField()[cellPatchIDBaffle];
			//Pout << "BaffleID = " << cellPatchIDBaffle << ", Surface area of baffle = " << surfA_Baffle << endl;
			const scalarField& surfA_Inj1 = mesh().magSf().boundaryField()[cellZoneIDInj1];
			const scalarField& surfA_Inj2 = mesh().magSf().boundaryField()[cellZoneIDInj2];
			scalarField surfA_Inj = surfA_Inj1;
			surfA_Inj.append(surfA_Inj2);
			const scalarField& surfA_Sink1 = mesh().magSf().boundaryField()[cellZoneIDSink1];
			const scalarField& surfA_Sink2 = mesh().magSf().boundaryField()[cellZoneIDSink2];
			scalarField surfA_Sink = surfA_Sink1;
			surfA_Sink.append(surfA_Sink2);
			//Pout << "surfA_Sink2 = " << surfA_Sink2 << ", surfA_Inj2 = " << surfA_Inj2 << endl;
			//----------CONSTANTS-----------//
			const double cm2m		= 1e-2;
			const double sm2scm		= 100*100;
			const double MW_CH3		= 15.03506/1000;//[kg/mol]
			const double MW_CH4		= 16.04303/1000;
			const double MW_H2O		= 18.01534/1000;
			const double MW_O2		= 31.9988/1000;
			const double kappa		= 2; //[W/m/K]
			double flux_Inj_Total = 0;
			int nCells = 0;
			ofstream spatial_fs;
			ofstream total_fs;		
			spatial_fs.open ("fvOptionsOut_spatial.dat");
			total_fs.open ("fvOptionsOut_total.dat", std::ofstream::out | std::ofstream::app);
			spatial_fs.precision(10);
			total_fs.precision(10);
			spatial_fs << "yLocation"<<"\t"<<"cell"<<"\t"<<"surfaceArea"<<"\t"<<"PO2feed"<<"\t"<<"PO2sweep"<<"\t"<< 
					"O2-flux[mol/qcm/s]"<<"\t"<<"CH4-flux[mol/qcm/s]"<<"\t"<<"CRR[mol(Ox)/qcm/s]"<<"\t"<<"heatFlux[W/s]"<<"\t"<<"cp(O2)[J/g/K]"<<"\n";
			
			//---------------INITIALIZE---------------//
			double PO2feed; double PO2sweep; double Tfeed; double Tsweep; double Temp; double PCH4;
			double CvII; double CvI; double JnO2 ; double cellFlowHeat;
			double surfA = 1e-6;
			//============================CALCULATE FLUX===============================//
			Pout << "Calculate O2 fluxes on feed side" << endl;
			forAll(cellListInj, i){
				nCells++;
				PO2feed	= pO2[cellListSink[i]]/101325.0; 	// make it atmospheres
				PO2sweep= pO2[cellListInj[i]]/101325.0;
				if(PO2feed > 1e-4 && PO2feed > PO2sweep){
					Tfeed	= T[cellListSink[i]];
					Tsweep	= T[cellListInj[i]];
					Temp	= (Tfeed+Tsweep)/2;
					PCH4	= pCH4[cellListInj[i]]/101325.0;
					//PCH4	= 0; //Make catalytic reaction meaningless
					//JO2
					CvII	= (CRR(k_CH4(Temp),PCH4)  +  pre_kr(Temp,D_v(Temp),k_f(Temp),PO2feed)  *  k_r(Temp))  / pre_CvII(Temp,D_v(Temp),k_f(Temp),PO2feed,PO2sweep);
					CvI		= (D_v(Temp)/(2*L) * CvII + k_r(Temp))  /  (D_v(Temp)/(2*L) + k_f(Temp)*pow(PO2feed,0.5));
					JnO2	= D_v(Temp)/(2*L)*(CvII-CvI);
					//other
					//CH4
					double fluxCH4		= CRR(k_CH4(Temp),PCH4);
					double cellFlowCH4	= fluxCH4*(surfA*sm2scm)*MW_CH4;
					//O2
					double fluxO2sweep	= JnO2-0.25*fluxCH4;
					if(fluxO2sweep < 0.0){
						//all O2 consumed by CH4
						fluxCH4 = 0.25*JnO2;
						fluxO2sweep = 0.0;
					}
					double cellFlowO2sweep	= fluxO2sweep*(surfA*sm2scm)*MW_O2;
					flux_Inj_Total = flux_Inj_Total + cellFlowO2sweep;
					//CH3
					double fluxCH3		= fluxCH4;			//vacancy concentration assumed to being constant!
					double cellFlowCH3	= fluxCH3*(surfA*sm2scm)*MW_CH3;
					//H2O
					double fluxH2O		= 0.5*fluxCH4;
					double cellFlowH2O	= fluxH2O*(surfA*sm2scm)*MW_H2O;
					//overall density calculation
					double allCellFlow	= -cellFlowCH4+cellFlowO2sweep+cellFlowCH3+cellFlowH2O;
					//overall enthalpy calculation
					double allEnthFlow	= (cellFlowO2sweep*cp_O2(Tfeed)+cellFlowCH3*cp_CH3(Tfeed)+
											cellFlowH2O*cp_H2O(Tfeed)-cellFlowCH4*cp_CH4(Tfeed))*(Tfeed)*1000; //*1000 because cp in [g], not in [kg]
					//---------heat conduction
					cellFlowHeat		= kappa*(Tsweep-Tfeed)/(L*cm2m)*surfA; //[W]
					if(cellFlowHeat < 0){0.1*cellFlowHeat;};
					//---------reaction enthalpy
					double cellFlowHcat = fluxCH4*(surfA*sm2scm)*100.215e3;
					//---------PLOT'N'WRITE---------//
					spatial_fs << C[cellListInj[i]].y()<<"\t"<<i<<"\t"<<surfA_Inj[i]<<"\t"<<PO2feed<<"\t"<<PO2sweep<<"\t"<<
							JnO2<<"\t"<<fluxCH4<<"\t"<<CRR(k_CH4(Temp),PCH4)<<"\t"<<cellFlowHeat<<"\t"<<cp_O2(Tfeed)<<"\n";
					//------------------------INJECTION (no calculation)----------------------------//
					switch (fieldI){ //'-=' needed here to get right input
						case 0: sourceI[cellListInj[i]]	-=  1*allCellFlow; break;//rho
						case 1: sourceI[cellListInj[i]]	-=  1*cellFlowO2sweep; break;//O2
						case 2: sourceI[cellListInj[i]]	-= -1*cellFlowCH4; break;//CH4, "-1" to indicate sink	
						case 3: sourceI[cellListInj[i]]	-=  1*cellFlowCH3; break;//CH3
						case 4: sourceI[cellListInj[i]]	-=  1*cellFlowH2O; break;//H2O	
						case 5: sourceI[cellListInj[i]] -=  1*allEnthFlow-cellFlowHeat-cellFlowHcat; break;//enthalpy
					};
				}; //end if
			}; // end injection loop
			//======CELL ITERATION SINK=================//			
			int nCellsF = 0;
			Pout << "Calculate O2 fluxes on sweep side" << endl;
			forAll(cellListSink, i)
			{
				nCellsF++;
				PO2feed		= pO2[cellListSink[i]]/101325.0;
				PO2sweep	= pO2[cellListInj[i]]/101325.0;
				if(PO2feed > 1e-4 && PO2feed > PO2sweep){
					Tfeed	= T[cellListSink[i]];
					Tsweep	= T[cellListInj[i]];
					Temp	= (Tfeed+Tsweep)/2;
					PCH4	= pCH4[cellListInj[i]]/101325.0;
					//PCH4	= 0;
					//---------OVERALL OXYGEN FLUX CALCULATION--------//
					double CvII		= (CRR(k_CH4(Temp),PCH4) + pre_kr(Temp, D_v(Temp),k_f(Temp), PO2feed)*k_r(Temp))/
							pre_CvII(Temp,D_v(Temp),k_f(Temp),PO2feed,PO2sweep);
					double CvI		= (D_v(Temp)/(2*L)*CvII+k_r(Temp))/(D_v(Temp)/(2*L)+k_f(Temp)*pow(PO2feed,0.5));
					double JnO2 	=  D_v(Temp)/(2*L)*(CvII-CvI);
					double cellFlowO2feed = JnO2*mag(surfA)*sm2scm*MW_O2;
					//---------heat flux
					cellFlowHeat	= kappa*(Tsweep-Tfeed)/(L*cm2m)*surfA;
					if(cellFlowHeat < 0){cellFlowHeat = 0.1*cellFlowHeat;};
					//---------SINK-----------------------------------//
					switch (fieldI){
						case 0: sourceI[cellListSink[i]] -= -1*cellFlowO2feed; break;
						case 1: sourceI[cellListSink[i]] -= -1*cellFlowO2feed; break;
						case 5: sourceI[cellListSink[i]] -= -1*cellFlowO2feed*cp_O2(Tfeed)*Tfeed*1000+cellFlowHeat; break;
					};
				};//end if
			};
			//=============================================//
			
			spatial_fs.close();  //close the spatial stream
			
			total_fs <<flux_Inj_Total<<"\n";
  			total_fs.close();
			
	
        #};

        codeSetValue
        #{
        #};

        code
        #{
            $codeInclude
            $codeCorrect
            $codeAddSup
            $codeSetValue
        #};
    }

    sourceTimeCoeffs
    {
    }
}



// ************************************************************************* //
