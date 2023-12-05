import numpy as np

from ..base import MultiGridEnv, MultiGrid
from ..objects import FreeDoor, Prey, GridAgentImaginedTraj


class PredatorPreyMultiGrid(MultiGridEnv):
    """
    Single room with red and blue doors on opposite sides.
    The red door must be opened before the blue door to
    obtain a reward.
    """

    mission = 'open the red door then the blue door'

    def __init__(self, config):
        self.size = config.get('grid_size')
        width = self.size
        height = self.size

        super(PredatorPreyMultiGrid, self).__init__(config, width, height)

    def _gen_grid(self, width, height, num_prey=4):
        """Generate grid without agents."""

        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the grid walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the preys
        self.preys = []
        for i in range(num_prey):
            prey = Prey(color='green',state=Prey.states.alive)
            # set a random position for each prey
            pos_x = self.np_random.randint(1, self.width - 1)
            pos_y = self.np_random.randint(1, self.height - 1)
            # if the position is already occupied, find another position
            while self.grid.get(pos_x, pos_y) is not None:
                pos_x = self.np_random.randint(1, self.width - 1)
                pos_y = self.np_random.randint(1, self.height - 1)
            self.grid.set(pos_x, pos_y, prey)
            prey.pos = np.asarray([pos_x, pos_y])
            self.preys.append(prey)

        return None

    def _reward(self):
        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _door_pos_to_one_hot(self, pos):
        p = np.zeros((self.width + self.height,))
        p[int(pos[0])] = 1.
        p[int(self.width + pos[1])] = 1.
        return p

    def gen_global_obs(self):

        obs = {
            'comm_act': np.stack([a.comm for a in self.agents],
                                 axis=0),  # (N, comm_len)
            'traj_comm_act': np.stack([a.traj_comm for a in self.agents],
                                axis=0),  # (N, comm_len)
            'env_act': np.stack([a.env_act for a in self.agents],
                                axis=0),  # (N, 1)
        }
        return obs

    def reset(self, is_evaluation=False):
        obs_dict = MultiGridEnv.reset(self)
        obs_dict['global'] = self.gen_global_obs()

        self.is_evaluation = is_evaluation
        if self.is_evaluation:
            self.agents_imagined_traj = []
            for i in range(len(self.agents)):
                for j in range(4):
                    imagined_step = GridAgentImaginedTraj(color=self.agents[i].color,is_evaluation=True)
                    # the initial position of the imagined step is the same as the agent
                    imagined_step.pos = self.agents[i].pos
                    self.agents_imagined_traj.append(imagined_step)

        self.agents_count = []
        for i in range(len(self.agents)):
            # generate a width*height empty 2d array for each agent
            self.agents_count.append(np.zeros((self.width, self.height)))
        return obs_dict

    def step(self, action_dict):
        obs_dict, rew_dict, _, info_dict = MultiGridEnv.step(self, action_dict)
        # update the count of each agent
        for i, agent in enumerate(self.agents):
            self.agents_count[i][agent.pos[0], agent.pos[1]] += 1
        # if all the preys are dead, the episode is success
        done = False
        success = False
        step_rewards = rew_dict['step_rewards']
        if all([prey.state == Prey.states.dead for prey in self.preys]):
            success = True
            done = True
            step_rewards += self._reward()
        else:
            success = False
        # if self.step_count >= self.max_steps, the episode is fail
        if self.step_count >= self.max_steps:
            done = True
            success = False

        agents_count_result = None
        if done:
            # if the episode is done, return agents_count/self.step_count
            # as an additional information
            agents_count_result = []
            for i in range(len(self.agents)):
                agents_count_result.append(
                    self.agents_count[i] / self.step_count)
        
        timeout = (self.step_count >= self.max_steps)

        obs_dict['global'] = self.gen_global_obs()
        rew_dict = {f'agent_{i}': step_rewards[i] for i in range(
            len(step_rewards))}
        done_dict = {'__all__': done or timeout}
        info_dict = {
            'done': done,
            'timeout': timeout,
            'success': success,
            # comm
            'comm': obs_dict['global']['comm_act'].tolist(),
            'traj_comm': obs_dict['global']['traj_comm_act'].tolist(),
            'env_act': obs_dict['global']['env_act'].tolist(),
            't': self.step_count,
            'agents_count': agents_count_result,
        }
        return obs_dict, rew_dict, done_dict, info_dict
