'''
Set up the game environment,which named qiuqiudazuozhan.
'''
from selenium.webdriver.common.action_chains import ActionChains
import cv2
import os
import requests
import numpy as np
import io
import base64
from PIL import Image
import time

class qiuqiu_env:
    def __init__(self):
        self.action_space={0: (200, 300,False), 1: (300, 300,False),
                           2: (400, 300,False), 3: (400, 400,False),
                           4: (400, 500,False), 5: (300, 500,False),
                           6: (200, 500,False), 7: (200, 400,False),
                           8: (200, 300, True), 9: (300, 300, True),
                           10:(400, 300, True), 11:(400, 400, True),
                           12:(400, 500, True), 13:(300, 500, True),
                           14:(200, 500, True), 15:(200, 400, True)}
        self.url="http://localhost:3000/get_data/"
        self.observation_size=(84,84,1) # (734, 600, 3)

    def step(self,driver,name,action_idx):
        agent_radius = self.get_radius(name)
        ### do action ###
        x,y,split_flag=self.action_space.get(action_idx)
        action=ActionChains(driver)
        can = driver.find_element_by_tag_name('canvas')
        action.move_to_element_with_offset(can, x, y)
        if split_flag:
            driver.find_element_by_css_selector("#split").click()
        action.click()
        action.perform()
        action.reset_actions()
        time.sleep(0.5)
        ### obs ###
        frame=self.get_screenshot(driver)
        obs=self._process_frame(frame)
        ### reward ###
        agent_radius_=self.get_radius(name)
        reward=agent_radius_-agent_radius
        ### done ###
        if np.mean(frame)<50:
            done=True
        else:
            done=False
        ### info ###
        info=None
        return obs,reward,done,info

    def reset(self,driver):
        self.click_play_btn_with_name(name='wang', driver=driver)
        driver.implicitly_wait(2)
        time.sleep(2)
        #obs = self.process_screenshot(driver)
        #return obs
    def stop_criterion(self,t):
        return None
    def _get_json_data(self,name):
        url = self.url + name
        response = requests.get(url)
        if response != None:
            json_data = response.json()
            # print("json_data", json_data)
            return json_data
        else:
            return None

    def click_play_btn_with_name(self,name, driver):
        driver.find_element_by_css_selector("#playerNameInput").send_keys(name)
        driver.find_element_by_css_selector("#startButton").click()

    def _process_frame(self,frame):
        img = np.reshape(frame, [734, 600, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_LINEAR)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)
    def get_radius(self,name):
        dict_data = self._get_json_data(name)
        if dict_data == None or len(dict_data) == 0:
            return
        agent_info = dict_data.get('visibleCells')
        agent_radius = agent_info[0].get('cells')[0].get('radius')
        return agent_radius
    def get_screenshot(self,driver):
        img_bin = driver.get_screenshot_as_base64()
        imgdata = base64.b64decode(img_bin)
        image = Image.open(io.BytesIO(imgdata))
        frame = np.array(image)
        return frame
    def process_screenshot(self,driver):
        frame=self.get_screenshot(driver)
        obs=self._process_frame(frame)
        return obs

