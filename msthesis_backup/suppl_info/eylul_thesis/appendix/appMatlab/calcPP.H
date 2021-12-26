const vectorField& C = mesh.C();

//volScalarField& N2  = Y[0];

forAll(C,i) //"i" represents the mesh element
{
	double YMWall = 0;
	for(int ii=0;ii< composition.species().size();ii++) //"ii" represents the species #
	{
		YMWall=YMWall+(Y[ii][i]/composition.W(ii));
	};

	/* MWmix is the molecular weight of the gas mixture: sum_ii{Y_ii*MW_ii} */
	MWmix[i] = 0;
	for(int ii=0;ii< composition.species().size();ii++) //"ii" represents the species #
	{
		MWmix[i]=MWmix[i]+Y[ii][i]/YMWall;
	};

	/* partial pressure calculated by: p_a = p * (Y_a/MW_a)/(sum_ii{Y_ii/MW_ii}) */
	for(int ii=0;ii< composition.species().size();ii++) //"ii" represents the species #
		{
			if(composition.species()[ii]=="CH4"){
				pCH4[i]  = p[i]*(Y[ii][i]/composition.W(ii))/YMWall;
			}
			if(composition.species()[ii]=="O2"){
				pO2[i]  = p[i]*(Y[ii][i]/composition.W(ii))/YMWall;
			}
		};
};


 
