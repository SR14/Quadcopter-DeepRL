import numpy as np
from physics_sim import PhysicsSim


class Takeoff():
    def __init__(self, target, i_pose=None, i_vel=None, i_avel=None, run_t=20.):
        """Initialize a Task object.
        Params
        ======
            i_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            i_vel: initial velocity of the quadcopter in (x,y,z) dimensions
            i_avel: initial radians/second for each of the three Euler angles
            run_t: time limit for each episode
            target: target/goal (z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(i_pose, i_vel, i_avel, run_t) 
        self.action_low = 0.
        self.action_high = 900.
        self.action_size = 1
        self.state_size = 2

        # Goal
        self.target = target

    def get_reward(self):
        disp = (self.sim.pose[2] - self.target) / self.start_dist
        disp = np.clip(disp, -1., 1.)
        vel = self.sim.v[2] / self.start_dist
        vel = np.clip(vel, -1., 1.)
        dist = np.absolute(disp)     
        speed = np.absolute(vel)
        
        distance_reward = 1 - dist
        speed_discount = 1 - speed
        
        reward = distance_reward * speed_discount  
        return disp, vel, reward

    def step(self, action):
        rotor_speeds = [self.action_high * action[0] for i in range(4)]
        done = self.sim.next_timestep(rotor_speeds) 
        disp, vel, reward = self.get_reward()
        next_state = [disp, vel]
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        start_disp = self.sim.pose[2] - self.target
        self.start_dist = np.absolute(start_disp)
        start_disp = start_disp / self.start_dist
        velocity = self.sim.v[2] / self.start_dist
        state = [start_disp, velocity]   
        return state