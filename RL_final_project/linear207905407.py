from policies import base_policy as bp
import numpy as np

LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.1
MIN_EPSILON = 0.01
EPSILON = 1
EPSILON_DECAY = 0.005
class Linear207905407(bp.Policy):

    features_vector = np.zeros(11)

    def cast_string_args(self, policy_args):
        policy_args['learn_rate'] = float(policy_args['learn_rate']) if 'learn_rate' in policy_args else LEARNING_RATE
        policy_args['discount_rate'] = float(policy_args['discount_rate']) if 'discount_rate' in policy_args else\
            DISCOUNT_RATE
        policy_args['min_epsilon'] = float(policy_args['min_epsilon']) if 'min_epsilon' in policy_args else MIN_EPSILON
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['epsilon_decay'] = float(policy_args['epsilon_decay']) if 'epsilon_decay' in policy_args else EPSILON_DECAY

        return policy_args

    def init_run(self):
        pass

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        correction = reward + self.discount_rate * self.getValue(new_state) - self.get_QValue_p(prev_state, prev_action)
        board, head = prev_state
        head_pos, direction = head
        current_position = head_pos.move(bp.Policy.TURNS[direction][prev_action])
        r = current_position[0]
        c = current_position[1]
        tmp_function = np.zeros(11)
        tmp_function[board[r, c] + 1] = 1
        self.features_vector = self.features_vector + self.learn_rate * (correction * tmp_function)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round %200 and self.epsilon>self.min_epsilon:
            self.epsilon -= self.epsilon_decay
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        chosen_act = self.get_QValue(new_state)

        return chosen_act

    def get_QValue(self, new_state):
        res = dict()
        for a in list(np.random.permutation(bp.Policy.ACTIONS)):
            res[a] = self.get_QValue_p(new_state, a)
        to_return = max(res, key=res.get)
        return to_return


    def get_QValue_p(self, state, action):
        board, head = state
        head_pos, direction = head
        next_position = head_pos.move(bp.Policy.TURNS[direction][action])

        r = next_position[0]
        c = next_position[1]
        tmp_function = np.zeros(11)
        tmp_function[board[r, c] + 1] = 1
        return self.features_vector.dot(tmp_function)

    def getValue(self, new_state):
        board, head = new_state
        head_pos, direction = head
        res = []
        for a in list(np.random.permutation(bp.Policy.ACTIONS)):
            next_position = head_pos.move(bp.Policy.TURNS[direction][a])
            r = next_position[0]
            c = next_position[1]
            tmp_function = np.zeros(11)
            tmp_function[board[r, c] + 1] = 1
            res.append(self.features_vector.dot(tmp_function))
        return max(res)
