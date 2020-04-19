
from policies import base_policy as bp
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

import subprocess

ACTION_DICT = {'L': 0, 'R': 1, 'F': 2}
DISCOUNT_RATE = 0.9
MIN_EPSILON = 0.001
EPSILON = 1
EPSILON_DECAY = 0.005
INPUT_DIM = 41 * 11

class Custom207905407(bp.Policy):
    model = Sequential()
    X = []
    def cast_string_args(self, policy_args):
        policy_args['discount_rate'] = float(policy_args['discount_rate']) if 'discount_rate' in policy_args else \
            DISCOUNT_RATE
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['min_epsilon'] = float(policy_args['min_epsilon']) if 'min_epsilon' in policy_args else MIN_EPSILON
        policy_args['epsilon_decay'] = float(policy_args['epsilon_decay']) if 'epsilon_decay' in policy_args else EPSILON_DECAY
        return policy_args

    def init_run(self):
        self.model.add(Dense(48, activation='relu', input_shape=(INPUT_DIM,)))
        self.model.add(Dense(10 , activation='relu'))
        self.model.add(Dense(3, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
            self.remember((prev_state, prev_action, reward, new_state))
            batch_size = len(self.X)
            X_inputs = np.zeros((batch_size,INPUT_DIM))
            y_outputs = np.zeros((batch_size,3))
            counter = 0
            for prev_state, prev_action,reward,new_state in self.X:
                prev_state_around = self.get_around(prev_state)
                X_inputs[counter] = prev_state_around
                prev_state_prediction = self.model.predict(np.array([prev_state_around]))
                new_state_around = self.get_around(new_state)
                prev_action_idx =ACTION_DICT[prev_action]
                t = reward + self.discount_rate * (np.amax(self.model.predict(np.array([new_state_around]))))
                prev_state_prediction[0][prev_action_idx] = t
                y_outputs[counter] = prev_state_prediction
                counter+=1
            self.X = []
            self.model.train_on_batch(X_inputs,y_outputs)



    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if round > 0 and round % 200 == 0:
            if self.epsilon> self.min_epsilon:
                 self.epsilon -= self.epsilon_decay

        if prev_action:
            self.remember((prev_state, prev_action, reward, new_state))

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        moves = self.get_around(new_state)
        total = self.model.predict(np.array([moves]))
        return bp.Policy.ACTIONS[np.argmax(total)]


    def remember(self,s):
        self.X.append(s)


    def get_around(self, state):
        board, head = state
        counter = 0
        around = np.zeros((41,11))
        stairs = 'FFFFF'
        turns = ['LFF','RFF']
        # gets whats to left and right of me
        for i in turns:
            tmp_head, tmp_direction = head
            for act in i:
                tmp_direction = bp.Policy.TURNS[tmp_direction][act]
                tmp_head = tmp_head.move(tmp_direction)
                around[counter][board[(tmp_head.pos)]+1] = 1
                counter+=1
        floor_head, floor_direction = head
        # gets what above in 5 steps and 3 to the left,right of each step.
        for f in stairs:
            floor_direction = bp.Policy.TURNS[floor_direction][f]
            floor_head = floor_head.move(floor_direction)
            around[counter][board[(floor_head.pos)] + 1] = 1
            counter += 1
            for i in turns:
                turn_head = floor_head
                turn_direction = floor_direction
                for act in i:
                    turn_direction = bp.Policy.TURNS[turn_direction][act]
                    turn_head = turn_head.move(turn_direction)
                    around[counter][board[(turn_head.pos)] + 1] = 1
                    counter += 1
        return around.flatten()

