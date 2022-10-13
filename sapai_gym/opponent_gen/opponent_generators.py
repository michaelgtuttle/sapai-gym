from sapai import Player, Team

from sapai_gym.ai import baselines
from sapai_gym import SuperAutoPetsEnv

# TODO : Wrap the ai to create a generator


def _do_store_phase(env: SuperAutoPetsEnv, ai):
    env.player.start_turn()

    while True:
        actions = env._avail_actions()
        chosen_action = ai(env.player, actions)
        env.resolve_action(chosen_action)

        if SuperAutoPetsEnv._get_action_name(actions[chosen_action]) == "end_turn":
            return


def _do_store_phase_DDQN(env: SuperAutoPetsEnv, ai, obs):
    env.player.start_turn()

    while True:
        action = ai.choose_action(obs)
        obs, reward, done, info = env.step(action)

        if action == 0: #SuperAutoPetsEnv._get_action_name(action) == "end_turn":
            return


def opp_generator(num_turns, ai):
    opps = list()
    env = SuperAutoPetsEnv(None, valid_actions_only=True, manual_battles=True)
    turn = 0
    while(turn <= num_turns):
        turn += 1
        obs = env.reset()
        while env.player.turn <= turn:
            _do_store_phase(env, ai)
        opps.append(Team.from_state(env.player.team.state))
    return opps


class opp_generator_DDQN():
    def __init__(self, ai):
        self.ai = ai
        self.env = SuperAutoPetsEnv(None, valid_actions_only=False, manual_battles=True)
    
    def __call__(self, num_turns):
        opps = list()
        turn = 0
        while(turn <= num_turns):
            turn += 1
            obs = self.env.reset()
            while self.env.player.turn <= turn:
                _do_store_phase_DDQN(self.env, self.ai, obs)
            opps.append(Team.from_state(self.env.player.team.state))
        return opps

class opp_gen():
    def __init__(self, ai):
        self.ai = ai
        self.env = SuperAutoPetsEnv(None, valid_actions_only=False, manual_battles=True)
    
    def __call__(self, num_turns):
        opps = list()
        obs = self.env.reset()
        while self.env.player.turn <= num_turns:
            _do_store_phase(self.env, self.ai, obs)
            opps.append(Team.from_state(self.env.player.team.state))
        return opps


def random_opp_generator(num_turns):
    return opp_generator(num_turns, baselines.random_agent)


def biggest_numbers_horizontal_opp_generator(num_turns):
    return opp_generator(num_turns, baselines.biggest_numbers_horizontal_scaling_agent)