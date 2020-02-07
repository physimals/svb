Command line usage
==================

Our implementation of Stochastic Variational Bayes includes a command line application
designed to be similar to Fabber_ (our implementation of analytic Variational Bayes).
The command line program is simply named ``svb``

Examples
~~~~~~~~

To fit the ASL data given in the FSL course we would use the following command line::

    svb --data=mpld_asltc.nii.gz --casl --plds=0.25,0.5,0.75,1.0,1.25,1.5 --slicedt=0.0452 \
        --tau=1.8 --repeats=8 \
        --mask=mpld_asltc_mask.nii.gz \
        --model=aslrest \
        --output=mpld_asltc_out 

.. _Fabber: https://fabber_core.readthedocs.io/
