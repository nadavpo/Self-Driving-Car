import glob
import os
import sys
import random
import time
import numpy as np
from tqdm import tqdm
# from PIL import Image
# import argparse
# from keras.callbacks import TensorBoard
# import tensorflow as tf
# import keras.backend.tensorflow_backend as backend
from threading import Thread
import cv2
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D
from tensorflow.keras.optimizers import Adam
from keras.initializers import he_normal
import math

try:
    sys.path.append(glob.glob(r'CARLA_0.9.5/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

MEMORY_SIZE = 5_000
BATCH_SIZE = 1
MIN_MEMORY_TO_TRAIN = BATCH_SIZE * 3
EPISODE_TO_RUN = 500
LEARNING_RATE = 0.001
UPDATE_TARGET_EVERY = 10
RENDER_EVERY = 50
SECONDS_PER_EPISODE = 10
AGGREGATE_STATS_EVERY = 10

MIN_REWARD = -100
GAMMA = 0.95
START_EPSILON = 1
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

IM_HEIGHT = 640
IM_WIDTH = 480

N_ACTIONS = 3
USE_RGB_CAMERA = True
USE_SEGMENTATION_CAMERA = True
USE_DEPTH_CAMERA = True

MODEL_NAME = "threeConvsGroups"


class DoubleQKN:
    def __init__(self, n_actions):
        self._MODEL_NAME = MODEL_NAME

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = self.create_network(n_actions)
        self.target_model = self.create_network(n_actions)
        self.target_model.set_weights(self.model.get_weights())

        self.training_initialized = False
        self.terminate = False

        self.current_episode = 0
        self.last_seen_episode = 0

    @staticmethod
    def create_network(n_actions):
        n_channels = int(USE_DEPTH_CAMERA) + int(USE_RGB_CAMERA) + int(USE_SEGMENTATION_CAMERA)
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(IM_HEIGHT, IM_WIDTH, n_channels),
                         activation='relu', kernel_initializer=he_normal(), padding='same'))
        # model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', kernel_initializer=he_normal(), padding='same'))
        # model.add(MaxPool2D(pool_size=(5, 5)))
        #
        # model.add(BatchNormalization())
        # model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', kernel_initializer=he_normal(), padding='same'))
        # model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', kernel_initializer=he_normal(), padding='same'))
        # model.add(BatchNormalization())
        # model.add(MaxPool2D(pool_size=(3, 3)))

        model.add(Flatten())
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(n_actions, activation='linear'))

        optimizer = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=optimizer)
        print(model.summary())
        return model

    def predict(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))

    def update_memory(self, data):  # data = (state, action, reward, next_state, done)
        self.memory.append(data)

    def train(self):
        if len(self.memory) < MIN_MEMORY_TO_TRAIN:
            return

        batch_data = random.sample(self.memory, BATCH_SIZE)

        curr_state = [x[0] for x in batch_data]
        curr_qs = self.model.predict(np.array(curr_state))

        next_state = [x[3] for x in batch_data]
        next_qs = self.target_model.predict(np.array(next_state))

        y_batch = []
        x_batch = []
        for index, data in enumerate(batch_data):
            state, action, reward, next_state, done = data

            new_q = reward
            if not done:
                future_reward = next_qs[index]
                new_q += GAMMA * np.max(future_reward)

            y = curr_qs[index]
            y[action] = new_q
            x_batch.append(state)
            y_batch.append(y)

        non_seen_episode = False
        if self.current_episode > self.last_seen_episode:
            non_seen_episode = True
            self.last_seen_episode = self.current_episode

        self.model.fit(x=np.array(x_batch), y=np.array(y_batch), batch_size=BATCH_SIZE, verbose=0,
                       shuffle=False, callbacks=None if non_seen_episode else None)

        if self.current_episode % UPDATE_TARGET_EVERY == 0:
            self.target_model.set_weights(self.model.get_weights())

    def start_training(self):
        # just to get things ready
        dummy_x = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        dummy_y = np.random.uniform(size=(1, 3)).astype(np.float32)
        self.model.fit(dummy_x, dummy_y, verbose=False, batch_size=1)
        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


class CarlaEnvironment:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        settings.no_rendering_mode = True
        self.world.apply_settings(settings)

        self.spawn_points = self.world.get_map().get_spawn_points()
        self.blueprint_library = self.world.get_blueprint_library()

        self.vehicle_bp = self.blueprint_library.filter('mustang')[0]
        self.vehicle_bp.set_attribute('color', '0,255,0')

        self.actor_list = []
        self.collision_hist = []
        self.last_frame_number = None
        self.SHOW_CAM = False
        self.episode_start = None
        self.rgb_cam_data = None
        self.depth_cam_data = None
        self.seg_cam_data = None
        self.vehicle = None
        self.collision_sensor_bp = None
        self.collision_sensor_location = None
        self.collision_sensor = None
        self.rgb_cam_bp = None
        self.rgb_cam_location = None
        self.rgb_cam = None
        self.seg_cam_bp = None
        self.seg_cam_location = None
        self.seg_cam = None
        self.depth_cam_bp = None
        self.depth_cam_location = None
        self.depth_cam = None

        self.rgb_data_ready = True
        self.depth_data_ready = True
        self.seg_data_ready = True

    def reset(self):
        self.collision_hist = []
        self.vehicle = None
        while self.vehicle is None:
            start_point = random.choice(self.spawn_points)
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, start_point)
            if self.vehicle is not None:
                self.actor_list.append(self.vehicle)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
        time.sleep(5.0)

        self.collision_sensor_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor_location = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.collision_sensor = self.world.spawn_actor(self.collision_sensor_bp, self.collision_sensor_location,
                                                       attach_to=self.vehicle)
        self.collision_sensor.listen(lambda data: self.process_collision_data(data))
        self.actor_list.append(self.collision_sensor)

        if USE_RGB_CAMERA:
            self.rgb_cam_bp = self.blueprint_library.find("sensor.camera.depth")
            self.rgb_cam_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
            self.rgb_cam_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
            self.rgb_cam_bp.set_attribute('fov', '110')
            self.rgb_cam_location = carla.Transform(carla.Location(x=1.5, z=2.4))
            self.rgb_cam = self.world.spawn_actor(self.rgb_cam_bp, self.rgb_cam_location, attach_to=self.vehicle)
            self.actor_list.append(self.rgb_cam)
            self.rgb_cam.listen(lambda data: self.process_rgb_data(data))
            self.rgb_data_ready = False

        if USE_SEGMENTATION_CAMERA:
            self.seg_cam_bp = self.blueprint_library.find("sensor.camera.semantic_segmentation")
            self.seg_cam_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
            self.seg_cam_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
            self.seg_cam_bp.set_attribute('fov', '110')
            self.seg_cam_location = carla.Transform(carla.Location(x=1.5, z=2.4))
            self.seg_cam = self.world.spawn_actor(self.seg_cam_bp, self.seg_cam_location, attach_to=self.vehicle)
            self.actor_list.append(self.seg_cam)
            self.seg_cam.listen(lambda data: self.process_segmentation_data(data))
            self.seg_data_ready = False

        if USE_DEPTH_CAMERA:
            self.depth_cam_bp = self.blueprint_library.find("sensor.camera.semantic_segmentation")
            self.depth_cam_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
            self.depth_cam_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
            self.depth_cam_bp.set_attribute('fov', '110')
            self.depth_cam_location = carla.Transform(carla.Location(x=1.5, z=2.4))
            self.depth_cam = self.world.spawn_actor(self.depth_cam_bp, self.depth_cam_location, attach_to=self.vehicle)
            self.actor_list.append(self.depth_cam)
            self.depth_cam.listen(lambda data: self.process_depth_data(data))
            self.depth_data_ready = False

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

        self.episode_start = time.time()

        self.world.tick()
        ts = self.world.wait_for_tick()

        self.last_frame_number = ts.frame_count

        while not (self.rgb_data_ready and self.seg_data_ready and self.depth_data_ready):
            time.sleep(0.001)

        self.rgb_data_ready = False
        self.seg_data_ready = False
        self.depth_data_ready = False

        return np.dstack(self.rgb_cam_data, self.depth_cam_data, self.seg_cam_data)

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        else:  # action == 2
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -1
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        self.world.tick()
        ts = self.world.wait_for_tick()

        if self.last_frame_number is not None:
            if ts.frame_count != self.last_frame_number + 1:
                print('frame skip!')

        self.last_frame_number = ts.frame_count

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        while not (self.rgb_data_ready and self.seg_data_ready and self.depth_data_ready):
            time.sleep(0.001)

        self.rgb_data_ready = False
        self.seg_data_ready = False
        self.depth_data_ready = False

        return np.dstack(self.rgb_cam_data, self.depth_cam_data, self.seg_cam_data), reward, done, None

    def process_collision_data(self, event):
        self.collision_hist.append(event)

    def process_rgb_data(self, data):
        i = np.array(data.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.rgb_cam_data = i3
        self.rgb_data_ready = True

    def process_depth_data(self, data):
        i = np.array(data.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 3))
        i3 = i2[:, :, :3]
        self.depth_cam_data = i3
        self.depth_data_ready = True

    def process_segmentation_data(self, data):
        i = np.array(data.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 3))
        i3 = i2[:, :, :3]
        self.seg_cam_data = i3
        self.seg_data_ready = True


def train():
    agent = DoubleQKN(N_ACTIONS)
    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.start_training, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    try:
        env = CarlaEnvironment()

        rewards = []
        min_reward = -100
        max_reward = 0
        average_reward = 0

        curr_epsilon = START_EPSILON
        # Initialize predictions - forst prediction takes longer as of initialization that has to be done
        # It's better to do a first prediction then before we start iterating over episode steps
        agent.predict(np.ones((IM_HEIGHT, IM_WIDTH, 3)))

        # Iterate over episodes
        for episode in range(EPISODE_TO_RUN):
            episode_reward = 0

            # Reset environment and get initial state
            current_state = env.reset()

            # Play for given number of seconds only
            while True:
                agent.current_episode = episode
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > curr_epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.predict(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, N_ACTIONS)

                new_state, reward, done, _ = env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory
                agent.update_memory((current_state, action, reward, new_state, done))

                current_state = new_state

                if done:
                    break

            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()

            print(f"episoe {episode}    reward: {episode_reward}")

            # Append episode reward to a list and log stats (every given number of episodes)
            rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = \
                    sum(episode_reward[-AGGREGATE_STATS_EVERY:])/len(episode_reward[-AGGREGATE_STATS_EVERY:])
                min_reward = min(episode_reward[-AGGREGATE_STATS_EVERY:])
                max_reward = max(episode_reward[-AGGREGATE_STATS_EVERY:])

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}'
                    f'min__{int(time.time())}.model')

            # Decay epsilon
            if curr_epsilon > MIN_EPSILON:
                curr_epsilon *= EPSILON_DECAY
                curr_epsilon = max(MIN_EPSILON, curr_epsilon)

        # Set termination flag for training thread and wait for it to finish
        agent.terminate = True
        trainer_thread.join()
        agent.model.save(
            f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}'
            f'min__{int(time.time())}.model')

    finally:
        agent.terminate = True
        trainer_thread.join()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--rgbcam', nargs='?', const=False, type=bool, default=False)
    # parser.add_argument('--depthcam', nargs='?', const=False, type=bool, default=False)
    # parser.add_argument('--segcam', nargs='?', const=False, type=bool, default=False)
    # args = parser.parse_args()

    train()
