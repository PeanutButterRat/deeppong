from dataclasses import dataclass

import numpy as np

import keras
import pygame
import random


# Pong constants.
SCREEN_HEIGHT, SCREEN_WIDTH = (160, 192)
SCREEN_CENTER_X, SCREEN_CENTER_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
SCREEN_FACTOR = 4
WINDOW_TITLE = 'DeepPong'
WINDOW_ICON_FILEPATH = 'assets/favicon.ico'

# Scoring.
SCORE_TO_WIN = 9
SCORE_PADDING = 10
SCORE_SIZE = 24
SCORE_FONT_FILEPATH = 'assets/font.ttf'
SCORE_SOUND_FILEPATH = 'assets/score.wav'
BOUNCE_SOUND_FILEPATH = 'assets/bounce.wav'
GOAL_PADDING = 8

# Center line.
DASH_LENGTH = 4
DASH_WIDTH = 2
GAP_LENGTH = 8

# Paddles and ball.
PADDLE_HEIGHT, PADDLE_WIDTH = (16, 2)
PADDLE_SPEED = 60
BALL_SIZE = 2
BALL_SPEED = 60

# Colors.
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Misc.
SOUND_VOLUME = 0.25
MAX_FPS = 30
FPS_PADDING = 10
FPS_SIZE = 16
IDEAL_DT = 1 / MAX_FPS
MAX_DT = IDEAL_DT * 1.5
MILLISECONDS_PER_SECOND = 1000


# Dataset generation constants.
RANDOM_SEED = 42

# Samples.
SCORE_COMBINATIONS = [(p1, p2) for p1 in range(SCORE_TO_WIN + 1) for p2 in range(SCORE_TO_WIN + 1) if not (p1 == SCORE_TO_WIN and p2 == SCORE_TO_WIN)]
STATES_PER_SCORE = 3
SAMPLES = STATES_PER_SCORE * len(SCORE_COMBINATIONS)
TIMESTEPS = 2
FEATURES = 10

# Directories.
DATA_DIRECTORY_NAME = 'data'
TRAINING_DIRECTORY_NAME = 'training'
BEST_MODEL_DIRECTORY_PATH = f'{TRAINING_DIRECTORY_NAME}/best'
SPREADSHEET_FILEPATH = f'{DATA_DIRECTORY_NAME}/data.csv'

# Model filepaths.
FCNN_SCORE_FILENAME = 'fnn_best_score.keras'
FCNN_LATENCY_FILENAME = 'fnn_best_latency.keras'
FCNN_BEST_FILEPATHS = [f'{BEST_MODEL_DIRECTORY_PATH}/{filename}' for filename in [FCNN_SCORE_FILENAME, FCNN_LATENCY_FILENAME]]

RNN_SCORE_FILENAME = 'rnn_best_score.keras'
RNN_LATENCY_FILENAMES = 'rnn_best_latency.keras'
RNN_BEST_FILEPATHS = [f'{BEST_MODEL_DIRECTORY_PATH}/{filename}' for filename in [RNN_SCORE_FILENAME, RNN_LATENCY_FILENAMES]]

CNN_SCORE_FILENAME = 'cnn_best_score.keras'
CNN_LATENCY_FILENAME = 'cnn_best_latency.keras'
CNN_BEST_FILEPATHS = [f'{BEST_MODEL_DIRECTORY_PATH}/{filename}' for filename in [CNN_SCORE_FILENAME, CNN_LATENCY_FILENAME]]


def clamp(x: float, low: float, high: float):
    '''Clamps x to the range [low, high].'''
    return max(low, min(x, high))


def colliding(x1: float, y1: float, w1: float, h1: float, x2: float, y2: float, w2: float, h2: float) -> bool:
    '''Determines if two rectangles are overlapping based on the top left coordinate and their dimensions.'''
    return not (x1 + w1 <= x2 or x1 >= x2 + w2 or y1 - h1 >= y2 or y1 <= y2 - h2)


@dataclass(eq=False)
class Paddle:
    '''Stores the state and controls for a paddle/player.'''

    x: float
    y: float
    up: int
    down: int
    score: int = 0


class Ball:
    '''Stores the state for the ball.'''

    def __init__(self):
        self.x = SCREEN_CENTER_X - BALL_SIZE / 2
        self.y = SCREEN_CENTER_Y + BALL_SIZE / 2
        self.vx = BALL_SPEED * (-1) ** random.randint(0, 1)
        self.vy = BALL_SPEED * random.uniform(-1, 1)


class Pong:
    '''The classic game of Pong.'''

    def __init__(self, sound_enabled=False):
        if not pygame.get_init():
            pygame.init()

        self.fps_font = pygame.font.Font(SCORE_FONT_FILEPATH, FPS_SIZE)
        self.current_renderer = 0
        self.renderers = [RasterizedRenderer(self)]
        self.renderer_names = ['Default (Rasterized)']
        self.sounds = []
        self.bounce_sound = None
        self.score_sound = None

        # Sound can be disabled if the environment doesn't support sound or the dataset is being generated.
        if sound_enabled:
            self.bounce_sound = pygame.mixer.Sound(BOUNCE_SOUND_FILEPATH)
            self.score_sound = pygame.mixer.Sound(SCORE_SOUND_FILEPATH)
            self.sounds = [self.bounce_sound, self.score_sound]
            self.mute()

        self.clock = pygame.time.Clock()        
        self.restart()

    def mute(self):
        '''Toggles sound for the game.'''

        for sound in self.sounds:
            volume = 0 if sound.get_volume() > 0 else SOUND_VOLUME
            sound.set_volume(volume)

    def update(self, dt, overrides={}):
        '''Updates the positions of all the game objects based on the amount of time that has passed since last frame (dt).'''

        if self.paused:
            return

        keys = pygame.key.get_pressed()

        # Move the paddles.
        for paddle in self.paddles:
            input = overrides.get(paddle.up, keys[paddle.up]) - overrides.get(paddle.down, keys[paddle.down])
            paddle.y += input * PADDLE_SPEED * dt
            paddle.y = clamp(paddle.y, PADDLE_HEIGHT, SCREEN_HEIGHT)

        # Update the ball's position.
        self.ball.y += self.ball.vy * dt
        self.ball.x += self.ball.vx * dt

        bounced = False
        scored = False

        # Vertical bounce.
        if self.ball.y >= SCREEN_HEIGHT:
            self.ball.y = SCREEN_HEIGHT - (self.ball.y - SCREEN_HEIGHT)
            self.ball.vy *= -1
            bounced = True
        elif self.ball.y - BALL_SIZE <= 0:
            self.ball.y += abs(self.ball.y - BALL_SIZE)
            self.ball.vy *= -1
            bounced = True

        # Check for goals if the game is in progress.
        if self.playing:
            for paddle in self.paddles:
                if colliding(paddle.x, paddle.y, PADDLE_WIDTH, PADDLE_HEIGHT, self.ball.x, self.ball.y, BALL_SIZE, BALL_SIZE):
                    bounced = True
                    self.ball.vx *= -1

                    if paddle == self.p1:
                        self.ball.x += paddle.x + PADDLE_WIDTH - self.ball.x
                    else:
                        self.ball.x -= self.ball.x + BALL_SIZE - paddle.x

                    offset = (self.ball.y - BALL_SIZE / 2) - (paddle.y - PADDLE_HEIGHT / 2)
                    standardized = clamp(offset / (PADDLE_HEIGHT / 2), -1, 1)
                    self.ball.vy = BALL_SPEED * standardized

            # Check to see if someone scored.
            if self.ball.x <= 0:
                scored = True
                self.p2.score += 1
                self.ball = Ball()
                self.ball.vx = abs(self.ball.vx)
            elif self.ball.x + BALL_SIZE >= SCREEN_WIDTH:
                scored = True
                self.p1.score += 1
                self.ball = Ball()
                self.ball.vx = -abs(self.ball.vx)

        # Otherwise the game is over: let the ball bounce freely against the goals without the paddles.
        elif self.ball.x <= 0:
            bounced = True
            self.ball.vx *= -1
            self.ball.x = abs(self.ball.x)
        elif self.ball.x + BALL_SIZE >= SCREEN_WIDTH:
            bounced = True
            self.ball.vx *= -1
            self.ball.x = SCREEN_WIDTH - (self.ball.x + BALL_SIZE - SCREEN_WIDTH)

        if bounced and self.bounce_sound is not None:
            self.bounce_sound.play()

        if scored and self.score_sound is not None:
            self.score_sound.play()

        # Check to see if the game is over (done every frame to make generating the dataset slightly easier).
        if max(self.p1.score, self.p2.score) == SCORE_TO_WIN:
            self.playing = False

    def restart(self):
        '''Restarts the game entirely.'''
    
        self.ball = Ball()
        self.p1 = Paddle(GOAL_PADDING, (SCREEN_HEIGHT + PADDLE_HEIGHT) / 2, up=pygame.K_w, down=pygame.K_s)
        self.p2 = Paddle(SCREEN_WIDTH - (GOAL_PADDING + PADDLE_WIDTH), (SCREEN_HEIGHT + PADDLE_HEIGHT) / 2, up=pygame.K_UP, down=pygame.K_DOWN)
        self.paddles = [self.p1, self.p2]
        self.playing = True
        self.paused = True
        self.show_fps = False

    def refresh(self):
        '''Renders and displays the next frame.'''

        renderer = self.renderers[self.current_renderer]
        surface = renderer.render()
        surface = pygame.transform.scale(surface, self.screen.get_size())
        self.screen.blit(surface, (0, 0))

        # Render the FPS and renderer name to evaluate the peformance in real-time.
        if self.show_fps:
            fps = round(self.clock.get_fps())
            text = f'FPS: {fps:02} - {self.renderer_names[self.current_renderer]}'
            fps = self.fps_font.render(text, False, WHITE)
            self.screen.blit(fps, (FPS_PADDING, FPS_PADDING))

        pygame.display.flip()

    def capture(self, screenshot=True):
        '''Returns the normalized game state and current frame.'''

        state = np.array([
            self.p1.x / SCREEN_WIDTH, self.p1.y / SCREEN_HEIGHT,
            self.p2.x / SCREEN_WIDTH, self.p2.y / SCREEN_HEIGHT,
            self.ball.x / SCREEN_WIDTH, self.ball.y / SCREEN_HEIGHT, self.ball.vx / BALL_SPEED, self.ball.vy / BALL_SPEED,
            self.p1.score / SCORE_TO_WIN, self.p2.score / SCORE_TO_WIN,
        ])

        return (state, self.renderers[0].render().copy()) if screenshot else state

    def tick(self):
        '''Ticks the in-game clock.'''

        milliseconds = self.clock.tick(MAX_FPS)
        dt = milliseconds / MILLISECONDS_PER_SECOND

        return clamp(dt, 0, MAX_DT)

    def run(self):
        '''Runs the game.'''

        pygame.display.set_caption(WINDOW_TITLE)
        pygame.display.set_icon(pygame.image.load(WINDOW_ICON_FILEPATH))
        self.screen = pygame.display.set_mode((SCREEN_WIDTH * SCREEN_FACTOR, SCREEN_HEIGHT * SCREEN_FACTOR))

        running = True
        dt = 0.0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.restart()
                    elif event.key == pygame.K_m:
                        self.mute()
                    elif pygame.K_1 <= event.key < pygame.K_1 + len(self.renderers):
                        self.current_renderer = event.key - pygame.K_1
                    elif event.key == pygame.K_f:
                        self.show_fps = not self.show_fps

            self.update(dt)
            self.refresh()

            dt = self.tick()

        pygame.quit()

    def add_renderer(self, renderer, name):
        self.renderers.append(renderer)
        self.renderer_names.append(name)


class Renderer:
    '''Base class for different types of 'rendering engines'.'''

    def __init__(self, pong):
        self.pong = pong
        self.frame = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    def render(self):
        '''Subclasses must implment this method.'''

        raise NotImplementedError


class RasterizedRenderer(Renderer):
    '''The traditional method of rendering using Pygame.'''

    def __init__(self, pong):
        super().__init__(pong)
        self.font = pygame.font.Font(SCORE_FONT_FILEPATH, SCORE_SIZE)

    def render(self, copy=False):
        self.frame.fill(BLACK)

        # Draw the left paddle's score.
        score = self.font.render(str(self.pong.p1.score), False, WHITE)
        self.frame.blit(score, ((SCREEN_WIDTH // 4) - (score.get_width() // 2), SCORE_PADDING))

        # Draw the right paddle's score.
        score = self.font.render(str(self.pong.p2.score), False, WHITE)
        self.frame.blit(score, ((3 * SCREEN_WIDTH // 4) - (score.get_width() // 2), SCORE_PADDING))

        # Draw the paddles.
        if self.pong.playing:  # Only if the game hasn't ended.
            for paddle in self.pong.paddles:
                rectangle = pygame.Rect(paddle.x, SCREEN_HEIGHT - paddle.y, PADDLE_WIDTH, PADDLE_HEIGHT)
                pygame.draw.rect(self.frame, WHITE, rectangle)

        # Draw the ball.
        square = pygame.Rect(self.pong.ball.x, SCREEN_HEIGHT - self.pong.ball.y, BALL_SIZE, BALL_SIZE)
        pygame.draw.rect(self.frame, WHITE, square)

        # Draw the center divider.
        for y in range(0, SCREEN_HEIGHT, DASH_LENGTH + GAP_LENGTH):
            dash = pygame.Rect(SCREEN_CENTER_X - DASH_WIDTH / 2, y, DASH_WIDTH, DASH_LENGTH)
            pygame.draw.rect(self.frame, WHITE, dash)

        return self.frame.copy() if copy else self.frame


class DeepLearningRenderer(Renderer):
    '''Base class for preding the next frame based on a deep-learning model.'''

    def __init__(self, pong, model):
        super().__init__(pong)
        self.model = keras.models.load_model(model)

    def render(self):
        frame = self.predict()

        # Reshape and size the prediction in the correct format.
        frame = frame.reshape((SCREEN_HEIGHT, SCREEN_WIDTH))
        frame = (frame * 255).astype(np.uint8)
        frame = np.stack([frame] * 3, axis=-1)
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        return surface

    def predict(self):
        raise NotImplementedError


class FullyConnectedRenderer(DeepLearningRenderer):
    '''Rendering with a simple, feed-forward network.'''

    def predict(self):
        state = self.pong.capture(screenshot=False).reshape(1, -1)
        return self.model.predict(state, verbose=0)


class RecurrentRenderer(DeepLearningRenderer):
    '''Rendering with a recurrent neural network based on multiple game states.'''

    def __init__(self, pong, model):
        super().__init__(pong, model)
        self.states = np.zeros((1, TIMESTEPS, FEATURES))

    def predict(self):
        state = self.pong.capture(screenshot=False).reshape(1, -1)
        
        # Forget the oldest game state and add the most recent one to the model input.
        self.states[0][:-1] = self.states[0][1:]
        self.states[0][-1] = state

        return self.model.predict(self.states, verbose=0)


class ConvolutionalRenderer(DeepLearningRenderer):
    '''Rendering with a convolutional neural network based on multiple game states.'''

    def __init__(self, pong, model):
        super().__init__(pong, model)
        self.states = np.zeros((1, TIMESTEPS, FEATURES))

    def predict(self):
        state = self.pong.capture(screenshot=False).reshape(1, -1)

        # Forget the oldest game state and add the most recent one to the model input.
        self.states[0][:-1] = self.states[0][1:]
        self.states[0][-1] = state

        # Reshape the input to have 1 channel for the CNN.
        input = self.states.reshape(1, TIMESTEPS, FEATURES, 1)

        return self.model.predict(input, verbose=0)


def main():
    '''The complete Pong application (models must be generated ahead of time).'''
    pong = Pong()

    # Add all the best trained models as renderers.
    pong.add_renderer(FullyConnectedRenderer(pong, FCNN_BEST_FILEPATHS[0]), 'FCNN (Best F1-Score)')
    pong.add_renderer(FullyConnectedRenderer(pong, FCNN_BEST_FILEPATHS[1]), 'FCNN (Best Latency)')
    pong.add_renderer(RecurrentRenderer(pong, RNN_BEST_FILEPATHS[0]), 'RNN (Best F1-Score)')
    pong.add_renderer(RecurrentRenderer(pong, RNN_BEST_FILEPATHS[1]), 'RNN (Best Latency)')
    pong.add_renderer(ConvolutionalRenderer(pong, CNN_BEST_FILEPATHS[0]), 'CNN (Best F1-Score)')
    pong.add_renderer(ConvolutionalRenderer(pong, CNN_BEST_FILEPATHS[1]), 'CNN (Best Latency)')

    pong.run()


# Include guard.
if __name__ == '__main__':
    main()
