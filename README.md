# Verified Exploration through Learned Models (VELM)

##  Introduction
We implement the Verified Exploration through Learned Models (VELM) in this artifact. Make sure Docker is installed on your system. 

For some of the commands, we list their estimated running time. These measurements are performed on a desktop computer with Intel I7-12700K CPU and 32GB RAM. The running time can very if tested on other machines.

## Installation
Follow the following instructions to build a new Docker image or use our provided image.

### Using the Provided Docker Image
1. Run the following command to download the image to local machine
    ```
    docker pull --platform=linux/x86_64 wyuning/velm
    ```

2. Start a new container from the provided image
    ```
    docker run --platform=linux/x86_64 -it wyuning/velm
    ```

    At this point, the container should be automatically in the VELM directory, please follow the instructions in the **Reproducibility Instructions** section.

### Build a New Docker Image
If the provided Docker image doesn't work for you for any reason, you can build a new image for you system.

1. Use the following command to build a image called velm (This process takes about 12 minutes)
    ```
    docker build -t velm .
    ```

2. Start a new container from this image
    ```
    docker run -it velm
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
```
python3 eval.py --start [x] --end [y] 
```

