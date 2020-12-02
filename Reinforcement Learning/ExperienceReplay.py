# Experience Replay
# Importing the libraries

import numpy as np
from collections import namedtuple, deque
import pyautogui
import pytesseract
from skimage.transform import resize
import time
import torch
from torch.autograd import Variable
import Utils


# Change the PyTesesseract Path as per your installation directory
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# Pre-process the Image to feed to the Neural Network
def preprocess(img):
    img_size = (height, width)
    img = resize(np.array(img), img_size)
    if grayscale:
        img = img.mean(-1, keepdims=True)
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32') / 255.
    return img


def do_action(action):
    if action == 0:  # Left
        pyautogui.mouseDown(x=920, y=820, button='left')
        pyautogui.mouseUp(x=800, y=820)
        pyautogui.moveTo(x=920, y=820)
    elif action == 1:  # Right
        pyautogui.mouseDown(x=920, y=820, button='left')
        pyautogui.mouseUp(x=1140, y=820)
        pyautogui.moveTo(x=920, y=820)
    elif action == 2:  # Up
        pyautogui.mouseDown(x=920, y=820, button='left')
        pyautogui.mouseUp(x=920, y=700)
        pyautogui.moveTo(x=920, y=820)
    elif action == 3:  # Down
        pyautogui.mouseDown(x=920, y=820, button='left')
        pyautogui.mouseUp(x=920, y=940)
        pyautogui.mouseUp(x=920, y=940)
        pyautogui.moveTo(x=920, y=820)
    elif action == 4:  # No move
        time.sleep(0.3)


def print_action(action):
    if action == 4:
        print("No move")
    elif action == 3:
        print("Down")
    elif action == 2:
        print("Up")
    elif action == 1:
        print("Right")
    else:
        print("Left")


# Defining one Step / Transition (Steps are saved in history and yielded back when the NStepProgress is iterated
Step = namedtuple('Step', ['state', 'action', 'reward', 'done', 'lstm'])

grayscale, height, width = True, 128, 128
game_dimensions = (675, 50, 570, 976)
saveme_dimensions = (870, 427, 165, 46)
score_dimensions = (1032, 200, 92, 35)
board_dimensions = (1096, 69, 133, 38)


# Making the AI Agent progress on several (n_step) steps
class NStepProgress:
    def __init__(self, ai, n_step):
        self.ai = ai  # AI Brain
        self.rewards = []
        self.n_step = n_step

    def __iter__(self):
        # Start the Game play
        pyautogui.click(x=1100, y=1000, clicks=1, button='left')
        pyautogui.moveTo(x=920, y=820)
        time.sleep(2)

        # Initialize state with the Game Screenshot
        state = preprocess(pyautogui.screenshot(region=game_dimensions))
        image = pyautogui.screenshot(region=game_dimensions)

        # Debugging
        # image_path = r"D:\Trail Ranger\SubwaySurfers\Temp\gameover1.png"
        # image.save(image_path)

        # History variable to store the Steps
        history = deque()

        la_actions = []
        la_states = []
        la_rewards = []

        is_done = True

        while True:
            if is_done:
                cx = Variable(torch.zeros(1, 256))
                hx = Variable(torch.zeros(1, 256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)

            action, (hx, cx) = self.ai(Variable(torch.from_numpy(np.array([state], dtype=np.float32))), (hx, cx))
            action = action[0][0]

            # Perform Action on the Game
            do_action(action)
            print_action(action)

            # Check game over state
            image1 = pyautogui.screenshot(region=saveme_dimensions)
            image2 = pyautogui.screenshot(region=score_dimensions)

            # Debugging - Use this for checking if Image dimensions are perfect
            # image1.save(image_path)
            # image = Utils.preprocess_image_for_ocr(image_path)

            # Perform text extraction
            # Recognizing Text to end Game
            text1 = pytesseract.image_to_string(image1, lang='eng', config='--psm 6')
            text2 = pytesseract.image_to_string(image2, lang='eng', config='--psm 6')

            r = 0

            if "M" in text1 or "Score" in text2:
                is_done = True
                r = -30  # Negative reward for the move which caused the game to end

                # Try popping the last 2 (state, action, reward) if Game over is detected
                if len(la_actions) >= 2:
                    la_actions.pop()
                    la_states.pop()
                    la_rewards.pop()
                    history.pop()

                if len(la_actions) >= 1:
                    action = la_actions.pop()
                    state = la_states.pop()
                    la_rewards.pop()
                    history.pop()

                print("Game End ! Penalty move -> ")
                print_action(action)

                # Handle Special case for Subway Surfer Emulator
                if "M" in text1:
                    # Wait for 1 sec and click Resume
                    time.sleep(1)
                    pyautogui.click(x=1100, y=1000, clicks=1, button='left')
                    time.sleep(1)

                pyautogui.click(x=1100, y=1000, clicks=1, button='left')
                pyautogui.moveTo(x=920, y=820)
                time.sleep(2)

            else:
                is_done = False
                r = 10  # Give Positive Reward for not losing / collecting coins!

                if action == 4:
                    r += 7  # Give Positive Reward for not making a move!

                next_state = preprocess(pyautogui.screenshot(region=game_dimensions))
                state = next_state

                la_actions.append(action)
                la_states.append(state)
                la_rewards.append(r)

            history.append(Step(state=state, action=action, reward=r, done=is_done, lstm=(hx, cx)))

            # -- Need to check for this snippet's use here --
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)
            # -- Need to check for this snippet's use here --

            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()

                self.rewards.append(sum(la_rewards))
                print("Rewards for this Step: " + str(self.rewards[-1]))
                print("Next Game started!")

                pyautogui.click(x=1100, y=1000, clicks=1, button='left')  # resumes the game
                pyautogui.moveTo(x=920, y=820)
                state = preprocess(pyautogui.screenshot(region=game_dimensions))

                la_actions = []
                la_states = []
                la_rewards = []
                history.clear()

    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps


