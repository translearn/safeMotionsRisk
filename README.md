# Safe Reinforcement Learning of Robot Trajectories in the Presence of Moving Obstacles 
[![IEEE RAL 2024](https://img.shields.io/badge/IEEE_RAL-2024-%3C%3E)](https://ieeexplore.ieee.org/document/10738380)
[![arXiv](https://img.shields.io/badge/arXiv-2411.05784-B31B1B)](https://arxiv.org/abs/2411.05784)
[![GitHub issues](https://img.shields.io/github/issues/translearn/safemotionsRisk)](https://github.com/translearn/safeMotionsRisk/issues/)<br>

This repository contains the code used for our paper *Safe Reinforcement Learning of Robot Trajectories in the Presence of Moving Obstacles.*

<div align='center'>
    <img src="https://github.com/user-attachments/assets/c20f450c-15f8-4639-ab85-7663688cb9e7" width="750"/>
</div>

## Installation

Our code is implemented and tested using Python 3.8. The required dependencies can be installed by running:

    pip install -r requirements.txt


## Trained networks 

We provide pretrained task networks, backup networks and risk networks. \
Rollouts for the task networks can be visualized in a simulator by running one of the commands below:  


### Reaching task

**Space environment, state-action-based risk**


```bash
python safemotions/evaluate.py --checkpoint=task_networks/reaching_task/space/state_action --no_exploration --visualize_risk --use_gui
```

**Ball environment, state-action-based risk**



```bash
python safemotions/evaluate.py --checkpoint=task_networks/reaching_task/ball/state_action --no_exploration --visualize_risk --use_gui
```


## Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
