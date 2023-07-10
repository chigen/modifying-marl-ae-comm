import numpy as np

from ..base import MultiGridEnv, MultiGrid
from ..objects import FreeDoor, COLORS


class TrafficJunctionMultiGrid(MultiGridEnv):
    """
    Available num_agents: 2 to 4
    Single room with different color doors on random sides of the room.
    The doors must be opened by the agent with the same color to 
    obtain a reward.
    If there is a collision between agents, there will be a penalty.
    """

    mission = 'each agent opens the same color door and avoid collision'

    def __init__(self, config):
        # size here should be 11*11
        self.size = config.get('grid_size')
        self.num_agents = config.get('num_agents')
        width = self.size
        height = self.size
        # aligned with agents' colors
        self.colors = ['red', 'blue', 'purple', 'orange']

        super(TrafficJunctionMultiGrid, self).__init__(config, width, height)

    def _gen_grid(self, width, height):
        """Generate grid without agents."""

        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the grid walls
        self.grid.wall_rect(0, 0, width, height)
        for i in range(1, 4):
            self.grid.horz_wall(x=i, y=1, length=4)
            self.grid.horz_wall(x=i, y=6, length=4)
        self.grid.horz_wall(x=5, y=1, length=3)
        self.grid.horz_wall(x=5, y=7, length=3)
        self.grid.horz_wall(x=7, y=1, length=3)
        self.grid.horz_wall(x=7, y=7, length=3)
        for i in range(8, 11):
            self.grid.horz_wall(x=i, y=1, length=4)
            self.grid.horz_wall(x=i, y=6, length=4)

        self.doors = []
        for i in range(self.num_agents):
            door = FreeDoor(color=self.colors[i],
                            state=FreeDoor.states.closed)
            self.doors.append(door)
        
        # self.red_door = FreeDoor(color='red', state=FreeDoor.states.closed)
        # self.blue_door = FreeDoor(color='blue', state=FreeDoor.states.closed)
        doors = self.doors
        self.np_random.shuffle(doors)

        # Add doors at different and random positions in different walls
        walls = []
        for i in range(self.num_agents):
            wall = self.np_random.randint(0, 4)
            while wall in walls:
                wall = self.np_random.randint(0, 4)
            walls.append(wall)
        
        for i in range(self.num_agents):
            pos = self.np_random.randint(4, 7)
            wall = walls[i]
            if wall == 0:
                self.grid.set(0, pos, doors[i])
                doors[0].pos = np.asarray([0, pos])
            elif wall == 1:
                self.grid.set(pos, 0, doors[i])
                doors[0].pos = np.asarray([pos, 0])
            elif wall == 2:
                self.grid.set(self.width - 1, pos, doors[i])
                doors[0].pos = np.asarray([self.width - 1, pos])
            elif wall == 3:
                self.grid.set(pos, self.height - 1, doors[i])
                doors[0].pos = np.asarray([pos, self.height - 1])
        
        # # Add a red/blue door at a random position in the left wall
        # pos = self.np_random.randint(1, self.size - 1)
        # self.grid.set(0, pos, doors[0])
        # doors[0].pos = np.asarray([0, pos])

        # # Add a red/blue door at a random position in the right wall
        # pos = self.np_random.randint(1, self.width - 1)
        # self.grid.set(self.width - 1, pos, doors[1])
        # doors[1].pos = np.asarray([self.width - 1, pos])

        return None

    def _reward(self):
        # TODO change the reward function
        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _door_pos_to_one_hot(self, pos):
        p = np.zeros((self.width + self.height,))
        p[int(pos[0])] = 1.
        p[int(self.width + pos[1])] = 1.
        return p

    def gen_global_obs(self):
        # concat door state and pos into a 1-D vector
        door_state = np.array([int(self.doors[i].is_open()) 
                               for i in range(self.num_agents)])
        # door_state = np.array([int(self.red_door.is_open()),
        #                        int(self.blue_door.is_open())])
        door_obs = np.concatenate([
            door_state,
            self._door_pos_to_one_hot(self.red_door.pos),
            self._door_pos_to_one_hot(self.blue_door.pos)])
        obs = {
            'door_obs': door_obs,
            'comm_act': np.stack([a.comm for a in self.agents],
                                 axis=0),  # (N, comm_len)
            'traj_comm_act': np.stack([a.traj_comm for a in self.agents],
                                axis=0),  # (N, comm_len)
            'env_act': np.stack([a.env_act for a in self.agents],
                                axis=0),  # (N, 1)
        }
        return obs

    def reset(self):
        obs_dict = MultiGridEnv.reset(self)
        obs_dict['global'] = self.gen_global_obs()
        return obs_dict

    def step(self, action_dict):

        red_door_opened_before = self.red_door.is_open()
        blue_door_opened_before = self.blue_door.is_open()

        obs_dict, _, _, info_dict = MultiGridEnv.step(self, action_dict)

        step_rewards = np.zeros((self.num_agents, ), dtype=np.float)

        red_door_opened_after = self.red_door.is_open()
        blue_door_opened_after = self.blue_door.is_open()

        if not red_door_opened_before and red_door_opened_after:
            red_door_opened_now = True
        else:
            red_door_opened_now = False

        done = False
        success = False
        if blue_door_opened_after:
            if red_door_opened_before:
                step_rewards += self._reward()
                success = True
                done = True
            else:
                done = True

        elif red_door_opened_after:
            if blue_door_opened_before:
                done = True

        timeout = (self.step_count >= self.max_steps)

        obs_dict['global'] = self.gen_global_obs()
        rew_dict = {f'agent_{i}': step_rewards[i] for i in range(
            len(step_rewards))}
        done_dict = {'__all__': done or timeout}
        info_dict = {
            'done': done,
            'timeout': timeout,
            'success': success,
            'comm': obs_dict['global']['comm_act'].tolist(),
            'env_act': obs_dict['global']['env_act'].tolist(),
            't': self.step_count,
            'red_door_opened_now': red_door_opened_now,
        }
        return obs_dict, rew_dict, done_dict, info_dict
