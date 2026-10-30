Outputs of main.py. Figures, cluster labels and k-means models are committed so
they can be inspected without running the pipeline.

<scenario>/snow_<scenario>.nc      Snow model output (snow_storage, snow_melt, rain)
                                   on the CESM grid. NOT committed (>100 MB each);
                                   main.py regenerates them from the input data.
47/global_analysis/                Global snow storage figures (16/47/150 Tg vs. control)
47/basin_analysis/                 River basin discharge figures (47 Tg vs. control)
47/clustering/, Control/clustering/
    elbow/                         Elbow method checkpoints (labels, models, inertias, plot)
    models/                        Final DTW k-means labels and model used in the analysis
