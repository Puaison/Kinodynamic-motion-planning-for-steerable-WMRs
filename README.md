# 🤖 Autonomous Mobile Robotics - 4-Wheel Steering Robot

An advanced robotics project implementing kinematic modeling, optimal path planning, and simulation for a mobile robot with four independent steering wheels. The main challenge tackled is the complex coordination of all four wheels simultaneously for smooth and efficient navigation.

## 📋 Table of Contents

- [Overview](#overview)
- [Technical Challenges](#technical-challenges)
- [Project Architecture](#project-architecture)
- [Key Features](#key-features)
- [Mathematical Foundation](#mathematical-foundation)
- [Implementation Details](#implementation-details)
- [Simulation Environment](#simulation-environment)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Documentation](#documentation)

## 🎯 Overview

This project addresses the complex problem of **autonomous navigation for a 4-wheel steering mobile robot**. Unlike conventional differential drive robots, our system features four independently steerable wheels, requiring sophisticated coordination algorithms to achieve smooth and optimal motion.

### 🚀 Project Goals

- **Kinematic Modeling**: Develop accurate mathematical models for 4-wheel steering dynamics
- **Optimal Path Planning**: Implement RRT* algorithm for generating collision-free optimal trajectories
- **Wheel Coordination**: Solve the challenging problem of synchronizing four independent steering mechanisms
- **Simulation**: Create a comprehensive testing environment for trajectory validation

## ⚡ Technical Challenges

### 1. **Multi-Wheel Coordination Complexity**
The primary challenge lies in coordinating four independent steering wheels simultaneously. Each wheel must:
- Maintain proper steering angles relative to the robot's instantaneous center of rotation
- Coordinate velocities to prevent wheel slippage
- Ensure kinematic constraints are satisfied throughout the motion

### 2. **Kinematic Constraints**
Four-wheel steering systems introduce complex kinematic relationships:
- **Ackermann Steering Geometry**: All wheels must rotate around a common instantaneous center
- **Velocity Coordination**: Linear and angular velocities must be properly distributed
- **Steering Angle Limitations**: Physical constraints on maximum steering angles

### 3. **Path Optimization**
Generating optimal paths requires considering:
- Obstacle avoidance in complex environments
- Smooth transitions between waypoints
- Kinematic feasibility of generated trajectories
- Computational efficiency for real-time applications

## 🏗️ Project Architecture

```
AutonomousMobileRobotics/
├── Model/                          # Kinematic & Dynamic Models
│   ├── model_creator.py           # Base and Dynamic model classes
│   └── old_model_creator.py       # Legacy implementations
├── Planning/                       # Path Planning Algorithms
│   └── planner.py                 # RRT* implementation with steering constraints
├── Control/                        # Trajectory Control & Execution
│   ├── optimal_trajectory.pkl     # Pre-computed optimal trajectories
│   ├── great_trajectory.pkl       # High-performance trajectory examples
│   └── test_RK4_*.py              # Runge-Kutta integration tests
├── World/                          # Simulation Environment
│   ├── occupancyGridMap.py        # Environment mapping and collision detection
│   ├── simulation.py              # Main simulation engine
│   └── simulation_with_traj.py    # Trajectory-based simulation
├── Test_uniciclo/                  # Unicycle Model Testing
│   ├── unycicle.py                # Single robot testing
│   └── multiple_unicycle.py       # Multi-robot scenarios
└── Utils/                          # Utilities & Constants
    ├── constants.py               # System parameters and configurations
    └── utils.py                   # Helper functions and utilities
```

## 📐 Mathematical Foundation

### **Kinematic Model**

The robot's kinematic model is based on the constraint that all four wheels must satisfy the no-slip condition:

$$\mathbf{q} = \begin{bmatrix} x \\ y \\ \theta \\ \phi_1 \\ \phi_2 \\ \phi_3 \\ \phi_4 \end{bmatrix}, \quad \mathbf{u} = \begin{bmatrix} v_1 \\ \omega \\ v_{\phi_1} \end{bmatrix}$$

Where:
- $(x, y, \theta)$: Robot pose in global coordinates
- $\phi_i$: Steering angle of wheel $i$
- $v_1$: Linear velocity of wheel 1
- $\omega$: Angular velocity of the robot
- $v_{\phi_1}$: Steering velocity of wheel 1

### **Kinematic Constraints**

The kinematic equations for the 4-wheel steering robot are derived from the no-slip condition:

$$\dot{x} = \cos(\theta + \phi_1) \cdot v_1 + \omega \cdot (P_{1x} \sin(\theta) + P_{1y} \cos(\theta))$$

$$\dot{y} = \sin(\theta + \phi_1) \cdot v_1 + \omega \cdot (P_{1y} \sin(\theta) - P_{1x} \cos(\theta))$$

$$\dot{\theta} = \omega$$

$$\dot{\phi_1} = v_{\phi_1}$$

### **Wheel Coordination Equations**

For proper 4-wheel steering, all wheels must rotate around a common Instantaneous Center of Rotation (ICR). The wheel positions in robot frame are:

$$\mathbf{P_1} = \begin{bmatrix} +0.24 \\ +0.19 \end{bmatrix}, \quad \mathbf{P_2} = \begin{bmatrix} -0.24 \\ +0.19 \end{bmatrix}, \quad \mathbf{P_3} = \begin{bmatrix} -0.24 \\ -0.19 \end{bmatrix}, \quad \mathbf{P_4} = \begin{bmatrix} +0.24 \\ -0.19 \end{bmatrix}$$

The kinematic constraint for each wheel $i$ to maintain no-slip condition:

$$\phi_i = \arctan\left(\frac{ICR_y - P_{i,y}}{ICR_x - P_{i,x}}\right) - \theta$$

### **Instantaneous Center of Rotation**

The ICR coordinates are computed from the first wheel's steering angle and velocity:

$$ICR_x = -\frac{P_{1y} + d \cos(\phi_1)}{\tan(\phi_1)}$$

$$ICR_y = P_{1x} + d \sin(\phi_1)$$

where $d$ is the wheel axis offset.

### **Dynamic Linearization**

The nonlinear system is linearized around the current operating point for control purposes:

$$\dot{\mathbf{q}} = \mathbf{f}(\mathbf{q}, \mathbf{u})$$

$$\dot{\mathbf{q}} \approx \mathbf{A}(\mathbf{q_0}) \Delta\mathbf{q} + \mathbf{B}(\mathbf{q_0}) \Delta\mathbf{u} + \mathbf{c}(\mathbf{q_0})$$

Where the Jacobian matrices are:

$$\mathbf{A} = \frac{\partial \mathbf{f}}{\partial \mathbf{q}} \bigg|_{\mathbf{q_0}, \mathbf{u_0}}, \quad \mathbf{B} = \frac{\partial \mathbf{f}}{\partial \mathbf{u}} \bigg|_{\mathbf{q_0}, \mathbf{u_0}}$$

$$\mathbf{c} = \mathbf{f}(\mathbf{q_0}, \mathbf{u_0}) - \mathbf{A} \mathbf{q_0} - \mathbf{B} \mathbf{u_0}$$

## 🛠️ Implementation Details

### **Model Creator (`Model/model_creator.py`)**
- **Base_Model Class**: Implements kinematic equations and linearization
- **Dynamic_Model Class**: Extends base model with dynamic constraints
- **Symbolic Computation**: Uses SymPy for exact mathematical derivatives

### **Path Planner (`Planning/planner.py`)**
- **RRT* Implementation**: Sampling-based optimal path planning
- **Node Class**: Tree node representation with state and cost information
- **Steering Function**: Connects nodes while respecting kinematic constraints
- **Collision Checking**: Integration with occupancy grid for obstacle avoidance

### **Simulation Engine (`World/simulation.py`)**
- **Real-time Animation**: Matplotlib-based visualization of robot motion
- **Physics Integration**: RK4 integration for smooth trajectory execution
- **Multi-wheel Visualization**: Individual wheel rendering with proper orientations

## 🌍 Simulation Environment

### **Occupancy Grid Mapping**
- **Configurable Environments**: Support for custom obstacle layouts
- **Inflation Algorithms**: Obstacle boundary inflation for safe navigation
- **Resolution Scaling**: Adjustable grid resolution for different scenarios

### **Trajectory Execution**
- **Smooth Interpolation**: Continuous trajectory generation between waypoints
- **Real-time Constraints**: Respects maximum velocities and accelerations
- **Visual Feedback**: Live visualization of robot state and planned path

### **Performance Metrics**
- **Path Optimality**: Cost analysis of generated trajectories
- **Execution Time**: Computational performance benchmarking
- **Constraint Violation**: Monitoring of kinematic constraint satisfaction

## ⚙️ Installation & Usage

### **Prerequisites**
```bash
pip install numpy matplotlib sympy pickle
```

### **Running Simulations**

1. **Basic Simulation**:
```bash
python World/simulation.py
```

2. **Path Planning**:
```bash
python Planning/planner.py
```

3. **Model Testing**:
```bash
python Control/test_RK4.py
```

### **Configuration**
Modify `Utils/constants.py` to adjust:
- Map dimensions and obstacles
- Robot physical parameters
- Planning algorithm settings
- Simulation visualization options

## 📚 Documentation

A comprehensive **technical report** (`Project_report.pdf`) is included, providing:

- **Detailed Mathematical Derivations**: Complete kinematic and dynamic formulations
- **Algorithm Analysis**: In-depth explanation of RRT* implementation and modifications
- **Experimental Results**: Performance analysis and comparison with alternative approaches

---

*This project demonstrates competency in robotics, control theory, and autonomous systems, tackling one of the most challenging problems in mobile robotics: coordinated multi-wheel steering for optimal autonomous navigation.*
