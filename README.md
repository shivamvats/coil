# [Optimal Interactive Learning on the Job via Facility Location Planning](https://optimal-interactive-learning.github.io/)
Welcome to our repository implementing COIL, as presented in:

Vats*, S., Zhao*, M., Callaghan, P., Jia, M., Likhachev, M., Kroemer, O., & Konidaris, G.D. (2025). **_Optimal Interactive Learning on the Job via Facility Location Planning_**. In Robotics: Science and Systems (RSS).

[<img src="https://img.shields.io/badge/arxiv-%23B31B1B.svg?&style=for-the-badge&logo=arxiv&logoColor=white" />](https://arxiv.org/pdf/2505.00490)


## Installation
We have tested our code on Ubuntu 20.04.

### Requirements
- [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)

### Installation Steps
1. Clone this repository and change directory into it:
    ```bash
    git clone git@github.com:shivamvats/coil.git 
    cd coil
    ```
2. Create a conda environment and activate it:
    ```bash
    conda env create -f environment.yaml
    conda activate coil-env
    ```
3. Install the local packages `robosuite` and `mimicgen`.
    ```bash
   pip install -e deps/robosuite
   pip install -e deps/mimicgen 
   ```
5. Install the `adaptive_teaming` (COIL) package:
    ```bash
    pip install -e .
    ```

---
## Planning with COIL
```bash
conda activate coil-env
```

The simplest way to run `COIL` is via the `run_interaction_planner.py` script:

```bash
python scripts/run_interaction_planner.py env=pick_place planner=fc_pref_planner task_seq.num_tasks=10 render=True
```
This script will simulate human-robot collaboration in our `mujoco` and `robosuite` based `pick_place` environment using the specified interaction planner. Results will be displaye on the terminal and automatically logged in a sub-directory in the `outputs` directory.

Parameters can be modified using the command line. For example, use `render=False` to disable rendering. All parameters and options are specified in `cfg/run_interaction_planner.yaml`. Please look at `make_planner` and `make_env` functions in `scripts/run_interaction_planner.py` for the list of supported planners and environment.

---
## Citation

If you use our work or code in your research, please cite our paper:
```latex
@inproceedings{vats2025optimal,
  title={Optimal Interactive Learning on the Job via Facility Location Planning},
  author={Vats, Shivam and Zhao, Michelle and Callaghan, Patrick and Jia, Mingxi and Likhachev, Maxim and Kroemer, Oliver and Konidaris, George},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025},
}
```
