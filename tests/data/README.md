# Test data

## What is in this folder

| File | What it is | Why it is here |
|---|---|---|
| `astronaut.jpg` | Portrait of astronaut Eileen Collins, NASA. Public domain (the same image as scikit-image's `data.astronaut()`). | The NudeNet model also detects faces. The model tests use its face class on this photo, so the real detector runs end to end without any explicit material. |
| `david_statue.mp4` | 3-second clip (360×640, 15 fps) of Michelangelo's David, a classical marble sculpture. It is a generated camera move over a CC0 photo by Jebulon ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Replica_Michelangelo%27s_David_black_background.jpg)). | Checks the real model and tracking on the region the tool is for. Because the camera move is generated, the region to censor is known exactly in every frame. |
| `david_statue.json` | That region (pubic hair and genitals, marked by hand on the source photo) as `[x1, y1, x2, y2]` for each of the 45 frames. | Ground truth for `test_statue_clip_is_covered_in_every_frame`. On the clip, `320n` plus tracking covers 97% of the region on average and at least 89% in every frame, with no mosaic anywhere else. |
| `make_statue_clip.py` | The script that builds the two files above from the source photo. | Reproducibility. |

## No explicit material in this repository

avmosaic does not include, host or distribute any explicit material, and the automated
tests do not need any. They use synthetic frames, a colour stand-in detector, a
public-domain portrait and a classical sculpture. Do not add explicit files to this
folder or anywhere else in the repository, and do not attach them to issues or pull
requests.

## Trying it on real footage (not provided)

We do not provide explicit test data. If you want to see how avmosaic does on real
footage, you can use a public research dataset such as
[The Image and video dataset for adult content detection in image and video](https://figshare.com/articles/dataset/The_Image_and_video_dataset_for_adult_content_detection_in_image_and_video/14495367?file=27771210)
(figshare).

> **18+, at your own risk.** Such datasets contain explicit adult content. Before
> downloading, read the dataset page, including its license and disclaimer. Use the data
> at your own risk and only where that is lawful, and keep it outside this repository.

## Checking accuracy on your own samples

Keep samples in a private folder outside the repository, then:

```bash
avmosaic frames video.mp4 -o ~/private/frames --every 2   # video -> images
avmosaic label ~/private/frames                           # mark what must be censored
avmosaic eval regions ~/private/frames                    # text report: coverage, spill, false alarms
AVMOSAIC_SAMPLES=~/private/frames pytest tests/test_samples.py -v   # the same check as tests
```

`avmosaic label` saves a `<name>.json` next to each image:
`{"regions": [[x1, y1, x2, y2], ...]}` for regions that must be censored, or
`{"regions": []}` for an image that must stay clean.
