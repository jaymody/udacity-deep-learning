import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
    
        self.n_state_params = 12
        self.state_size = self.action_repeat * self.n_state_params
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
       
    def get_reward(self, done):
        """Returns reward based on current state."""
        
        ### Implementation 1 ###
#         # Rewards agent for moving as close to max speed as possible
#         cur_velocity = np.sum(i**2 for i in self.sim.v) ** 0.5
#         vel = 0.2 * abs(cur_velocity - self.max_v)
#         vel = np.tanh(vel)
        
#         # Rewards the agent for being stable in regards to its angular velocity
#         cur_angular_velocity = np.sum(i**2 for i in self.sim.angular_v) ** 0.5
#         angle = 0.002 * abs(cur_angular_velocity)
#         angle = np.tanh(angle)
        
#         # Rewards agent for being closer and closer to the objective
#         pos = 0.02 * abs(self.sim.pose[:3] - self.target_pos).sum()
#         pos = 2 * np.tanh(pos)

        ### Implementaion 2 ###
#         angle = 0.05 * abs(np.sum(self.sim.angular_v))
#         pos = 0.25 * abs(np.sum(self.sim.pose[:3] - self.target_pos))
#         vel = abs(np.sum(np.subtract(self.target_pos - self.sim.pose[:3] , self.sim.v)))


        # Current Velocity
        cur_velocity = np.sum(i**2 for i in self.sim.v) ** 0.5

        pos = 0.25 * abs(np.sum(self.sim.pose[:3] - self.target_pos)) # rewards agent for being close to target
        vel = 0.25 * abs(np.sum(self.sim.v)) # rewards agent for having a low velocity
        angle = 0.05 * abs(np.sum(self.sim.angular_v)) # rewards agent for being stable
        elevation = 0.125 * abs(self.sim.pose[3]) # rewards agent for staying away from the ground
        
        print("{:7.3f} || {:7.3f} || {:7.3f} || {:7.3f} || {:7.3f} || {}".format\
              (cur_velocity, pos, vel, angle, elevation, self.sim.pose[:3]))
        
        # the 1 rewards the agent for staying alive
        return 1. - pos - vel - angle - elevation

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done)
            
            # MY CODE #
            curr_state = []
            curr_state.append(self.sim.pose)
            curr_state.append(self.sim.v[:3])
            curr_state.append(self.sim.angular_v[:3])
            curr_state = np.concatenate(curr_state)
            ###########
            
            pose_all.append(curr_state)
        
        next_state = np.concatenate(pose_all)
        
        
        # Punsished agent for terminating due to crashing on the ground
        if (abs(np.sum(next_state[2::self.n_state_params] < 0.01))):
            done = True
            reward += -250
            print("CRASHED")
        # Rewards agent for reaching the goal
        elif (abs(next_state[0] - self.target_pos[0]) < 0.5 and
            abs(next_state[1] - self.target_pos[1]) < 1 and
            abs(next_state[2] - self.target_pos[2]) < 5):
            reward += 300
            done = True
            print("GOAL REACHED")
        
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        
        # MY CODE #
        curr_state = []
        curr_state.append(self.sim.pose)
        curr_state.append(self.sim.v[:3])
        curr_state.append(self.sim.angular_v[:3])
        ##########
        
        state = np.concatenate(curr_state * self.action_repeat) 
        return state