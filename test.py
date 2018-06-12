# -*- coding: utf-8 -*-
#
# author: oldj <oldj.wu@gmail.com>
#
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import numpy as np
import multiprocessing
import threading
import requests
import time
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import base64
import os
from io import BytesIO
import io

def capture(url, save_fn="capture.png"):
    browser = webdriver.Chrome()  # Get local session of firefox
    browser.set_window_size(1200, 900)
    browser.get(url)  # Load page
    browser.execute_script("""
    (function () {
      var y = 0;
      var step = 100;
      window.scroll(0, 0);

      function f() {
        if (y < document.body.scrollHeight) {
          y += step;
          window.scroll(0, y);
          setTimeout(f, 50);
        } else {
          window.scroll(0, 0);
          document.title += "scroll-done";
        }
      }

      setTimeout(f, 1000);
    })();
  """)

    for i in range(30):
        if "scroll-done" in browser.title:
            break
        time.sleep(1)

    browser.save_screenshot(save_fn)
    browser.close()

def get_json_data(name,url):
    url=url+name
    response=requests.get(url)
    if response !=None:
        json_data=response.json()
        return json_data
    else:
        return None
def click_play_btn_with_name(name, driver):
    driver.find_element_by_css_selector("#playerNameInput").send_keys(name)
    driver.find_element_by_css_selector("#startButton").click()
    print("Playing..")

def test1():
    name = 'worker_' + '0'
    options = webdriver.ChromeOptions()
    options.add_argument("--window-size=600,800")
    options.add_argument("--allow-file-access-from-files")
    options.add_argument("--disable-infobars")
    driver = webdriver.Chrome(chrome_options=options)
    driver.get('http://localhost:3000')
    driver.implicitly_wait(2)
    print(driver.find_element_by_css_selector("#startMenuWrapper").value_of_css_property('max-height'))
    click_play_btn_with_name(name, driver)
    driver.implicitly_wait(2)
    #time.sleep(1)

    dire_dict = {0: (200, 300), 1: (300, 300), 2: (400, 300), 3: (400, 400),
                 4: (400, 500), 5: (300, 500), 6: (200, 500), 7: (200, 400)}
    print(driver.find_element_by_css_selector("#startMenuWrapper").value_of_css_property('max-height'))
    action = ActionChains(driver)
    i=0
    #driver.save_screenshot(name + '_%d' % i)
    img_bin=driver.get_screenshot_as_base64()
    imgdata=base64.b64decode(img_bin)
    image = Image.open(io.BytesIO(imgdata))
    #cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


    while True:
        idx=i%8
        x,y=dire_dict.get(idx)
        can = driver.find_element_by_tag_name('canvas')
        action.move_to_element_with_offset(can, x, y)
        action.click()
        action.perform()
        action.reset_actions()
        i+=1


def test2():

    response = requests.get('http://pic35.nipic.com/20131121/2531170_145358633000_2.jpg')  # 将这个图片保存在内存
    # 将这个图片从内存中打开，然后就可以用Image的方法进行操作了
    image = Image.open(BytesIO(response.content))
    # 得到这个图片的base64编码
    ls_f = base64.b64encode(BytesIO(response.content).read())
    # 打印出这个base64编码
    print(type(ls_f))
    imgdata = base64.b64decode(ls_f)
    print(imgdata)
    file = open('3.jpg', 'wb')
    file.write(imgdata)
    # 关闭这个文件
    file.close()


if __name__ == '__main__':
    test1()



