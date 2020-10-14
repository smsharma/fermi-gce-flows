A subset of the Fermi-LAT public data for use with NPTFit:

https://github.com/bsafdi/NPTFit

The data here is for use with the ipython example notebooks provided with the
main code. Details of the files provided are given below. All files are provided 
as npy files binned as nside=128 HEALPix maps.

For the full public data, see:

http://fermi.gsfc.nasa.gov/ssc/data/access/

===============================
========= Fermi Data ==========
===============================
Number of files: 3
File naming: fermidata_<datatype>.npy
Description:
    - counts: a subset of the Fermi data [photon counts]. In detail:
        - Energy Range: 2-20 GeV;
        - Time Period: Aug 4, 2008 to July 7, 2016 (413 weeks);
        - Event Class: UltracleanVeto (1024);
        - Event Type: PSF3 (32);
        - Quality Cuts: DATA_QUAL==1 && LAT_CONFIG==1; and
        - Max Zenith Angle: 90 degrees
    - exposure: the exposure map associated with the above data [cm^2 s]
    - pscmask: mask of all point sources in the 3FGL at 1 degree; the mask also
               includes large extended objects like the LMC [boolean array]
               Point sources are masked at 95% containment according to the
               Fermi point spread function at 2 GeV

===============================
==== Background Templates =====
===============================
Number of files: 6
File naming: template_<model>.npy
Description:
    - dif: the p6v11 model of diffuse Galactic emission
    - iso: isotropic emission
    - psc: all point sources in the 3FGL
    - bub: the Fermi Bubbles
    - gce: a line of sight integrated NFW squared profile, used as a model
           for the galactic centre excess
    - dsk: a thin disk used as a spatial template for galactic point sources
Details:
    All templates have been exposure corrected, so that they are maps of counts
    not flux.
