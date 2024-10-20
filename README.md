# Safe Reinforcement Learning of Robot Trajectories in the Presence of Moving Obstacles 

This repository contains the code used for our paper *Safe Reinforcement Learning of Robot Trajectories in the Presence of Moving Obstacles.*

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
