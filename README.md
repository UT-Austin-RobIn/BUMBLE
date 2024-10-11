# BUMBLE: Unifying Reasoning and Acting with Vision-Language Models for Building-wide Mobile Manipulation
![Image](assets/BUMBLE_pull_fig.svg)
[Rutav Shah](https://shahrutav.github.io/), [Albert Yu](https://scholar.google.com/citations?user=ZzURcb4AAAAJ&hl=en), [Yifeng Zhu](https://zhuyifengzju.github.io/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/)<sup>†</sup>, [Roberto Martín-Martín](https://robertomartinmartin.com/)<sup>†</sup>  
<sup>†</sup> Equal Advising

[[Paper]](https://arxiv.org/abs/2410.06237)    [[Project Website]](https://robin-lab.cs.utexas.edu/BUMBLE/)

# Setup

### Installing ROS
Follow the guide provided at the official ROS wiki to install [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu) on your system.

### Setting up Python Environment
```bash
git clone git@github.com:UT-Austin-RobIn/BUMBLE.git
conda create -y -n bumble python=3.9
conda activate bumble
cd BUMBLE
python -m pip install -r requirements.txt
python -m pip install -r rospy_requirements.txt
git clone https://github.com/mjd3/tracikpy.git
python -m pip install tracikpy/
python -m pip install -e .
```

### Installing GSAM:
Download the SAM-HQ weights from the [original repository](https://github.com/SysCV/sam-hq?tab=readme-ov-file#model-checkpoints). We use the weights of the model:  [ViT-B HQ-SAM model](https://drive.google.com/file/d/11yExZLOve38kRZPfRx_MRxfIAKmfMY47/view)  
Set the environemnt variable: 
```
export SAM_CKPT_PATH=/path/to/sam_hq_vit_b.pth
```
Installing GSAM:
```bash
git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything/GroundingDINO && python setup.py build && python setup.py install
cd ../../
python -m pip install -e Grounded-Segment-Anything/segment_anything/
```

### Setting up VLM API
Set the environment variable `OPENAI_API_KEY` to your OpenAI API key.

### Setting up Building Occupancy Map
We use Pal's rosservice (`change_map`) to set the map for the Tiago robot (See `set_floor_map` inside `bumble/tiago/ros_restrict`). You should add similar functionality to set the 2D occupancy map for your ROS packages programmatically.  

After setting the map, configure the landmark locations for the GoToLandmark skill. You can add your own landmarks or use the provided [landmarks](https://utexas.box.com/s/el33g5od55rku2qheddhbmembossis6p).  
Example of landmark image structure: 
```
bumble/tiago/skills/landmark_images/{BUILDING_NAME}_landmark_images{FLOOR_NUM}/{BUILDING_NAME}{FLOOR_NUM}_{LANDMARK_INDEX}.jpg
```  
Note: The provided landmarks correspond to university buildings used in the experiments and are mapped to the relevant building occupancy maps.  

# Usage
To run the main script (`rw_eval.py`) for BUMBLE, run the following command:
```bash
python rw_eval.py --run_vlm --add_selection_history --add_past --exec --method ours --floor_num <FLOOR_NUM> --bld <BUILDING_NAME> --eval_id 2 --n_eval 1 --run_dir <PATH_TO_EXP_DIR>
```
To prevent long-term memory from being added, remove the --add_past flag.  

# Citation
```bash
@article{shah2024bumble,
    title={BUMBLE: Unifying Reasoning and Acting with Vision-Language Models for Building-wide Mobile Manipulation},
    author={Rutav Shah and Albert Yu and Yifeng Zhu and Yuke Zhu and Roberto Martín-Martín},
    journal={arXiv preprint arXiv:2410.06237},
    year={2024},
}
```

# [Acknowledgements](Acknowledgements.md)
