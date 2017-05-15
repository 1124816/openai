import gym
import random
import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dense, Dropout, Activation
from statistics import median, mean
from collections import Counter
from keras import optimizers

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 90
initial_games = 5000

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(200):
            env.render()

            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            if done:
                break


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    print(training_data_save)
    np.save('saved.npy',training_data_save)

    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

def neural_network_model(input_size):
    model = Sequential()
    #network = input_data(shape=[None, input_size, 1], name ='input')

    model.add(Dense(units=128, activation='relu', input_shape=(input_size,)))
    model.add(Dropout(0.3))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
    return model

def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data])
    y = [i[1] for i in training_data]
    print(X[-1])
    print(y[-1])

    if not model:
        print(len(X[0]))
        model = neural_network_model(input_size = len(X[0]))

    model.fit(X, y, epochs=70, batch_size=32)
    print(model.evaluate(X, y, verbose=1, batch_size=32))
    return model
model = train_model(initial_population())



def next_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(100):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            if len(prev_observation)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_observation)[0])
                #print(np.argmax(model.predict(prev_observation.reshape(-1,len(prev_observation),1))[0]))
            env.render()
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    print(training_data_save)
    np.save('saved.npy',training_data_save)

    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


scores = []
choices = []
for each_game in range(100):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    while True:

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(np.array([prev_obs]))[0])
        #print(action)
        choices.append(action)
        #env.render()

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
#print(score_requirement)

#model = train_model(next_population(), model)
#
#scores = []
#choices = []
#for each_game in range(100):
#    score = 0
#    game_memory = []
#    prev_obs = []
#    env.reset()
#    for _ in range(goal_steps):
#
#        if len(prev_obs)==0:
#            action = random.randrange(0,2)
#        else:
#            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
#            print(np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0]))
#
#        choices.append(action)
#
#        new_observation, reward, done, info = env.step(action)
#        prev_obs = new_observation
#        game_memory.append([new_observation, action])
#        score+=reward
#        if done: break
#
#    scores.append(score)
#
#print('Average Score:',sum(scores)/len(scores))
#print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
#print(score_requirement)
#
#
#
