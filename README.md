# Verified Exploration through Learned Models (VELM)

##  Introduction
We implement the Verified Exploration through Learned Models (VELM) in this artifact. Make sure Docker is installed on your system. 

For some of the commands, we list their estimated running time. These measurements are performed on a MacBook Pro Intel I5 CPU (4 cores) and 16GB RAM. The running time can very if tested on other machines.

## Installation
Follow the following instructions to build a new Docker image or use our provided image. Note this process will create a Docker container named `velm_container` that might have the the same name with your existing containers.

### Using the Provided Docker Image
1. Run the following command to download the image to local machine
    ```
    docker pull --platform=linux/x86_64 wyuning/velm
    ```

2. Start a new container from the provided image
    ```
    docker run --platform=linux/x86_64 -it --name velm_container wyuning/velm
    ```

    At this point, the container should be automatically in the VELM directory, please follow the instructions in the **Reproducibility Instructions** section.

### Build a New Docker Image
If the provided Docker image doesn't work for you for any reason, you can build a new image for you system.

1. Use the following command to build a image called velm (This process takes about 20 minutes)
    ```
    docker build -t velm .
    ```

2. Start a new container from this image
    ```
    docker run -it --name velm_container velm
    ```

    At this point, the container should be automatically in the VELM directory, please follow the instructions in the **Reproducibility Instructions** section.



## Code Structure
In this section, we brief introduce major components of our code.

1. VELM:
    - The `train` function in the `main_stable_baseline.py` file implements the VELM algorithm (algorithm 1 in the paper). This function includes the logic of learning symbolic environment model and tranining the RL agent using SAC algorithm.

    - The `compute_safe_agent` function in the `main_stable_baseline.py` file implements the Shield algorithm (algorithm 2 in the paper). This function is called by the `train` function to compute a sheilded controller
    
    - The `DynamicsLearner.run` function in the `dynamics/learn.py` file implements part of the Approximate (algorithm 3 in the paper). This function is called by the `compute_safe_agent` function to learn a time-varying linear controller.

    - The `run_lagrange` function in the `verification/VEL/lagrange_lib.py` file implements the Lagragian optimization described in the paper. This function is called by the `compute_safe_agent` function to perform iterative Lagrangian optimization.

2. Controller Verification:
    - The controller verification source code for all benchmarks are in the `POLAR_Tool/examples` directory. The controller verification code will be called to compute the specification violation loss for a given time-varying linear controller.
    - The compiled controller verification executable files are in the `verification/executables/` directory.

3. Logs: The `logs` directory is initially empty. The log files generated will be stored in this location.

4. Provided logs and figure
    - The `provided_logs` directory contains logs that are generated using the same process described in this document.
    - The `provided_figure.png` is the figured obtained by running the visualization script using the provided logs.


## Reproducibility Instructions

### Running Provided Evaluation script
We provide an evaluation script named `eval.py` that will run benchmarks in batches.
```
python3 eval.py --row <x>
```
where x can be 1, 2, 3, meaning running the benchmarks in the x-th row in the figure 4 of the paper..
Here we list the estimated time for running each row
| Row    | Time |
| -------- | ------- |
| 1  | 2 hours    |
| 2 |  2 hours    |
| 3    | 1 hour   |


The evaluation script will also generate a figure named `eval.png`. This figure can be copied out of the Docker container to the current directory using the following command. Note that this command should be run outside the container, for example in another terminal.
```
docker cp velm_container:/VELM/eval.png .
``` 


### Running Individual benchmarks
To run an individual benchmark, please run
```
python3 main_stable_baseline.py <benchmark name>
```
where benchmark name can be `pendulum`, `obstacle`, `obstacle_mid`, `road_2d`, `cartpole`, `cartpole_move`, `cartpole_swing`, `lalo`, `car_racing`, `acc`

After running invidual benchmark, the figure can be updated with
```
python3 eval.py
```

