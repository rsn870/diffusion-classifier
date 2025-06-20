## Zero-Shot Classification with Merger-Aware Guidance

Our primary contribution is an updated `eval_prob_adaptive.py` script that integrates our merger-aware analysis. This replaces the original's brute-force search over timesteps (`--to_keep`, `--n_samples`) with a more principled and efficient workflow.

### Recommended Workflow: Automatic Interval Finding

This workflow first performs a one-time analysis on a dataset to find the optimal classification windows and then uses them for evaluation.

```bash
python eval_prob_adaptive.py \
  --dataset cifar10 --split test \
  --prompt_path prompts/cifar10_prompts.csv \
  --auto-find-intervals \
  --analysis-dataset-path /path/to/your/cifar10/root \
  --weighting-strategy truncated-inverse-snr
```

**What this command does:**
1.  **Analyzes**: It loads the full dataset from `--analysis-dataset-path` and computes the "merger time" (`t_stop,k`) for each class.
2.  **Classifies**: It then proceeds to classify images from the `--dataset`, restricting the evaluation to the most discriminative time window for each class, as determined by the `--weighting-strategy` and the computed merger times.

### Evaluating on Your Own Dataset

1.  **Create a Prompt File**: Prepare a `.csv` file with `prompt`, `class_name`, and `class_idx` columns. The `class_name` must match the folder names in your analysis dataset (e.g., `tench` for ImageNet, `cat` for CIFAR-10).
2.  **Run the Classifier**: Use the command above, changing `--dataset`, `--prompt_path`, and `--analysis-dataset-path` to match your use case.
3.  **Choose a Weighting Strategy**: The `--weighting-strategy` flag selects the importance weighting `w(t)` from our paper. `truncated-inverse-snr` is recommended as it consistently performs best.

### New Command-Line Arguments in `eval_prob_adaptive.py`

| Argument | Type | Description |
|:---|:---|:---|
| `--auto-find-intervals` | `flag` | Enables the automatic analysis phase to find merger times before classifying. |
| `--analysis-dataset-path` | `str` | **Required** with auto-find. Path to the dataset (with class-named subfolders) used for the analysis. |
| `--weighting-strategy` | `str` | The importance weighting to use: `uniform`, `inverse-snr`, or `truncated-inverse-snr` (default). |
| `--truncated-start-time`| `int` | The start time `t_start,k` for the `truncated-inverse-snr` strategy. Default is `20`. |
| `--num-noise-seeds` | `int` | The number of noise seeds `N` to average over for each score. Default is `250`. |

*(Note: The original arguments like `--to_keep` and `--n_samples` are no longer used by the merger-aware workflow.)*

## Standard ImageNet Classification with DiT
This repository maintains compatibility with the original class-conditional DiT evaluation scripts.

### Additional Installations
Within the project folder, clone the original DiT repository:
```bash
git clone https://github.com/facebookresearch/DiT.git
```

### Running DiT-based Classification
First, save a consistent set of noise that will be used for all image-class pairs:
```bash
python scripts/save_noise.py --img_size 256
```
Then, compute the epsilon-prediction error for each class on each image. For example, to evaluate class 207:
```bash
python eval_prob_dit.py  --dataset imagenet --split val \
  --noise_path noise_256.pt --randomize_noise \
  --batch_size 32 --cls 207 --t_interval 4 --extra dit256 --save_vb
```
This process is computationally expensive and should be parallelized across multiple machines for all 1000 classes.

Finally, compute the accuracy using the saved error files:
```bash
python scripts/print_dit_acc.py data/imagenet_dit256
```


## Citation

If you use our merger-aware methods or analysis tools in your research, please cite our work. We also ask that you cite the original "Diffusion Classifier" paper, on which this repository is based.

```bibtex
@article{Ramachandran2025CrossFluctuation,
  title={Revealing Sampling Dynamics in Diffusion Models via Cross-Fluctuation Phase Transitions},
  author={Sai Niranjan Ramachandran and Manish Krishan Lal and Suvrit Sra},
  year={2025},
  journal={arXiv preprint},
}
```

```bibtex
@misc{li2023diffusion,
      title={Your Diffusion Model is Secretly a Zero-Shot Classifier}, 
      author={Alexander C. Li and Mihir Prabhudesai and Shivam Duggal and Ellis Brown and Deepak Pathak},
      year={2023},
      eprint={2303.16203},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
