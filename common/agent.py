class Environment:

    def get_num_states(self):
        pass

    def allowed_actions(self, state_id):
        pass

    def state_distribution(self, state_id, action_id):
        pass

    def reward(self, state_id, action_id, next_state_id):
        pass


class Agent:
    def evaluate_policy(self):
        pass

    def update_policy(self):
        pass
