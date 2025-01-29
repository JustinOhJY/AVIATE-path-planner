import subprocess
import time

# Commands for the simulation
commands = [
    # Terminal 1: Launch Gazebo Simulation
    "gnome-terminal -- bash -c 'eval \"$(conda shell.bash hook)\" && conda activate aerial_robotics && source ~/aerial_robotics_ws/devel/setup.bash && roslaunch robowork_minihawk_gazebo minihawk_playpen.launch; exec bash'",

    # Terminal 2: Start Ardupilot SITL
    "gnome-terminal -- bash -c 'eval \"$(conda shell.bash hook)\" && conda activate aerial_robotics && cd ~/aerial_robotics_ws/ardupilot && ./Tools/autotest/sim_vehicle.py -v ArduPlane -f gazebo-minihawk --model gazebo-quadplane-tilttri --console --map --wipe; exec bash'",

    # Terminal 3: Launch RViz Visualization
    "gnome-terminal -- bash -c 'eval \"$(conda shell.bash hook)\" && conda activate aerial_robotics && source ~/aerial_robotics_ws/devel/setup.bash && rviz -d ~/aerial_robotics_ws/src/aerial_robotics/robowork_minihawk_launch/config/minihawk_SIM.rviz; exec bash'",

    # Terminal 4: Launch MAVROS Node
    "gnome-terminal -- bash -c 'eval \"$(conda shell.bash hook)\" && conda activate aerial_robotics && source ~/aerial_robotics_ws/devel/setup.bash && ROS_NAMESPACE=\"minihawk_SIM\" roslaunch robowork_minihawk_launch vehicle1_apm_SIM.launch; exec bash'",

    # Terminal 5: Set Mode to GUIDED
    "gnome-terminal -- bash -c 'eval \"$(conda shell.bash hook)\" && conda activate aerial_robotics && source ~/aerial_robotics_ws/devel/setup.bash && rosservice call /minihawk_SIM/mavros/set_mode \"custom_mode: \\'GUIDED\\'\"; exec bash'"
]

# Execute the commands
for cmd in commands:
    subprocess.run(cmd, shell=True)
    time.sleep(2)
