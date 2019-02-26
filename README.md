# metadetect-sims

simulations for the metadetect paper

# Notes for Usage

This is not a user-friendly place.

Before doing anything, install the local package in editable model

```bash
$ pip install -e .
```

Then change into the directory with the simulation you would like to run.

## Running at BNL

To run a sim at BNL, you need to make the condor submit scripts via

```bash
$ python make_condor_job_script.py
```

Then you can run the incremental submission script that @esheldon wrote

```bash
$ ~esheldon/python/bin/condor-incsub *.desc
```

Finally, you can analyze the sims using

```bash
python do_fit_condor.py
```
