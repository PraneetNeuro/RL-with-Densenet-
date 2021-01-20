import tensorflow as tf
import gym
import numpy as np
import random
from tqdm import tqdm


class Environment:
    def __init__(self, environment_id, games_to_collect_data_from, steps_for_success, score_req, save):
        self.env = gym.make(environment_id)
        self.init_state = self.env.reset()
        self.training_data = []
        self.games_to_collect_data_from = games_to_collect_data_from
        self.steps_for_success = steps_for_success
        self.score_required = score_req
        self.save = save

    def populate_training_data(self):
        scores = []
        accepted_scores = []
        for _ in tqdm(range(self.games_to_collect_data_from)):
            score = 0
            game_memory = []
            prev_observation = []
            for _ in range(self.steps_for_success):
                action = random.randrange(0, 2)
                observation, reward, done, info = self.env.step(action)
                if len(prev_observation) > 0:
                    game_memory.append([prev_observation, action])
                prev_observation = observation
                score += reward
                if done: break
            if score >= self.score_required:
                accepted_scores.append(score)
                for data in game_memory:
                    output = []
                    if data[1] == 1:
                        output = [0, 1]
                    elif data[1] == 0:
                        output = [1, 0]
                    self.training_data.append([data[0], output])
            self.init_state = self.env.reset()
            scores.append(score)
        training_data_save = np.array(self.training_data)
        if self.save:
            np.save('training_data.npy', training_data_save)


class NeuralNet:
    def __init__(self, training_data, input_size, env, epochs=5):
        self.training_data = training_data
        self.input_size = input_size
        self.network = tf.keras.Sequential()
        self.neural_network_model()
        self.epochs = epochs
        self.train_model()
        self.env = env

    def neural_network_model(self):
        self.network.add(tf.keras.layers.Input(shape=[self.input_size]))
        self.network.add(tf.keras.layers.Dense(128, activation='relu'))
        self.network.add(tf.keras.layers.Dropout(0.8))
        self.network.add(tf.keras.layers.Dense(256, activation='relu'))
        self.network.add(tf.keras.layers.Dropout(0.8))
        self.network.add(tf.keras.layers.Dense(128, activation='relu'))
        self.network.add(tf.keras.layers.Dropout(0.8))
        self.network.add(tf.keras.layers.Dense(2, activation='softmax'))
        self.network.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        X = np.array([i[0] for i in self.training_data]).reshape(-1, len(self.training_data[0][0]))
        y = [i[1] for i in self.training_data]
        self.network.fit(np.asarray(X), np.asarray(y))
        self.network.save('model_{}'.format(self.epochs))

    def test_model(self, games=10):
        for each_game in range(games):
            prev_obs = []
            self.env.env.reset()
            for _ in range(self.env.steps_for_success):
                self.env.env.render()
                if len(prev_obs) == 0:
                    action = random.randrange(0, 2)
                else:
                    action = np.argmax(self.network.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
                new_observation, reward, done, info = self.env.env.step(action)
                prev_obs = new_observation
                if done:
                    print('Solved {} / {} successfully'.format(each_game, games))
                    break


env = Environment('CartPole-v0', 10000, 200, 50, True)
env.populate_training_data()
model = NeuralNet(env.training_data, 4, env)
model.test_model()
