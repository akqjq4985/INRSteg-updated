# Implicit Steganography Beyond the Constraints of Modality

[Project Page](https://inrsteg.github.io/) | [Paper](https://arxiv.org/abs/2312.05496)

Sojeong Song*, Seoyun Yang*, Chang D. Yoo, Junmo Kim
KAIST, South Korea, * denotes equal contribution

This is the official implementation of the paper "Implicit Steganography Beyond the Constraints of Modality", which is accepted at ECCV 2024.

## Get started
Make pytorch docker container, and run `setup.sh` file.
```bash
sh setup.sh <CONTAINER_NAME>
```

## How to run
Prepare data for secret and cover in `/data/cover` and `/data/secret/`, and set hyperparameters in `config.yaml`.

```bash
python main.py
```

## Update Plan
- [X] Hide and reveal secret data
- [ ] Evaluation on reconstructed results
- [ ] Steganalysis analysis
- [ ] Permutation application

## SIREN
We're using the excellent [siren](https://github.com/vsitzmann/siren/tree/master?tab=readme-ov-file) as baseline.

### License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
- This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/)

### Copyright
- All codes on this page are copyrighted by Sojeong Song published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License. You must attribute the work in the manner specified by the author. You may not use the work for commercial purposes, and you may only distribute the resulting work under the same license if you alter, transform, or create the work.

## Citation
If you find our work useful in your research, please cit:

```
@InProceedings{10.1007/978-3-031-73016-0_17,
author="Song, Sojeong
and Yang, Seoyun
and Yoo, Chang D.
and Kim, Junmo",
title="Implicit Steganography Beyond the Constraints of Modality",
booktitle="Computer Vision -- ECCV 2024",
year="2025",
publisher="Springer Nature Switzerland",
}
```

## Contact
If you have any questions, please feel free to email the authors.
