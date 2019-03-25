Contact: Ryan Dale, ryan.dale@nih.gov

This Jupyter notebook can be used to reproduce the particle tracking analysis
in Vogt et al (2019) "Anchoring mouse cortical granules in the egg cortex ensures
timely trafficking to the plasma membrane for post-fertilization exocytosis",
Nature Communications.

For more information on using Jupyter notebooks, see the
[documentation](https://jupyter-notebook.readthedocs.io/en/stable/).

We use [`conda`](https://conda.io/docs/) to manage the requirements in
a reproducible fashion. If you already have the Anaconda Python distribution,
you already have `conda`. Otherwise, please install
[Miniconda](https://conda.io/miniconda.html), which is a minimal installation.

Once `conda` is installed, run the following from a terminal:

```bash
conda env create -n particle-tracking --file full_environment.yaml
```

This will create an environment with all software required to run this
notebook. Then you can activate the environment with:

```
source activate particle-tracking
```

and then start the notebook with 

```
jupyter notebook manuscript-figures.ipynb
```

*This notebook was run on Linux using the full `conda` environment specified in
`full_environment.yaml`. That file lists the exact version of all packages
installed at the time the figures were generated, and serves as a more archival
record of the packages. It is likely that this notebook will run on MacOS and
Windows, but this is untested. Success relies on availability of packages on
those platforms.*