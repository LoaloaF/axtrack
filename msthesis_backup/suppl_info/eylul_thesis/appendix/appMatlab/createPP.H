//Create partial pressure fields
Info<< "Creating partial pressure fields.\n" << endl;

volScalarField MWmix
    (
        IOobject
        (
            "MWmix",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("MWmix", dimMass/dimMoles, 0.0)
    );
volScalarField pCH4
    (
        IOobject
        (
            "pCH4",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("pCH4", dimPressure, 0.0)
    );

volScalarField pO2
    (
        IOobject
        (
            "pO2",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("pO2", dimPressure, 0.0)
    );

/*basicSpecieMixture gasMixture
    (
        IOobject
        (
            "gasMixture",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        thermo.composition()
    );*/
