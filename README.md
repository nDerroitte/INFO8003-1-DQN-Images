# INFO8003-1 - Optimal decision making for complex problems
## Assignment 3
This file comments on how to use the code done in the second assignment. The first step is to install the packages needed for the code.
### Intsall
To create the video, `opencv` has been used. In the second part, `skitlearn` was used.
To install the related packages, one can just run the install bash script :
```sh
$ ./install.sh
```
### Run the code
In order to run the code, one should simply use the following :
```sh
$ python3 run.py <options>
```
When called without option, the code runs  with the default value of the arguments. In order to change that, one can use the following parameters:
* `--nb_episodes` **int** : Number of episodes used (>0).
* `--discount_factor` **float** : Discount factor (gamma).
* `--video` **{0,1}** : 1 if the user wants a video of the simulation. 0 otherwise.
* `--plot` **{0,1}** : 1 if the user wants plot of the simulation. 0 otherwise.
