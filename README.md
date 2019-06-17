# ETH3D SLAM Evaluation Program #

This tool is used for evaluating SLAM methods in the [ETH3D benchmark](https://www.eth3d.net/).

Usage:

```
ETH3DSLAMEvaluation \
    ground_truth.txt \
    estimated_trajectory.txt \
    <rgb.txt or depth.txt> \
    [--sim3] \
    [--max_interpolation_timespan t] \
    [--all_visualizations] \
    [--write_estimated_trajectory_ply]
```

If you use the dataset for research, please cite our paper:

[Thomas Schöps, Torsten Sattler, Marc Pollefeys, "BAD SLAM: Bundle Adjusted Direct RGB-D SLAM", CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Schops_BAD_SLAM_Bundle_Adjusted_Direct_RGB-D_SLAM_CVPR_2019_paper.html).