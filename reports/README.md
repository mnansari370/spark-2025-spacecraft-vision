# Reports

This folder holds the scripts that compute the metrics and generate the plots and tables used in our report and presentation. Nothing here is needed to run training or inference.

```
reports/
├── jobs/       # SLURM scripts for the reporting jobs
├── latest/     # Generated output
│   ├── figures/
│   └── tables/
└── scripts/    # Metric computation and plotting
```

The figures and tables under `latest/` are the actual generated results (COCO detection metrics, per class AP, segmentation IoU per class, and the pixel count statistics).
