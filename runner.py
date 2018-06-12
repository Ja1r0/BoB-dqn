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

#url = "http://192.168.9.243:3000/get_data/"
url="http://localhost:3000/get_data/"
dangerous_cells = []
dangerous_distance = 350
emergency_distance = 260
food_distance = 450
food_cells = []
p_random_move = 0.1

def work(name):
    name = "worker_"+str(name)
    options = webdriver.ChromeOptions()
    #options.binary_location = '/usr/local/share/chromedriver'
    # options.add_argument('disable-gpu')
    # options.add_argument('window-size=620x180')
    options.add_argument("--window-size=600,800")
    options.add_argument("--allow-file-access-from-files")
    options.add_argument("--disable-infobars")
    if not name == "worker_0":
        options.add_argument("--headless")
    # options.add_argument("--headless")
    #driver = webdriver.Chrome(executable_path='/home/jairo/local/anaconda3/envs/serpent/bin/chromedriver',chrome_options=options)
    driver=webdriver.Chrome(chrome_options=options)
    #driver.get('http://192.168.9.243:3000/')
    driver.get('http://localhost:3000')
    driver.implicitly_wait(2)
    # if driver.find_element_by_css_selector("#startButton"):
    #print(driver.find_element_by_css_selector("#startMenuWrapper").value_of_css_property('max-height'))
    click_play_btn_with_name(name, driver)
    driver.implicitly_wait(2)
    time.sleep(1)

    while True:
        play(driver,name)
        #driver.save_screenshot(name+'_%d'%i)
        if driver.find_element_by_css_selector("#startMenuWrapper").value_of_css_property('max-height') == '1000px':
            #print('died')
            click_play_btn_with_name("miao", driver)
            driver.implicitly_wait(2)
            #print('restart')
            time.sleep(2)

def click_play_btn_with_name(name, driver):
    driver.find_element_by_css_selector("#playerNameInput").send_keys(name)
    driver.find_element_by_css_selector("#startButton").click()
    #print("Playing..")

def random_direction():
    # 8 directions
    dire_dict = { 0: (200, 300), 1: (300, 300), 2: (400, 300), 3: (400, 400),
                 4: (400, 500), 5: (300, 500), 6: (200, 500), 7: (200, 400) }
    return dire_dict.get(random.randint(0, 7))

def play(driver,name,url="http://localhost:3000/get_data/",p_random_move = 0.1):
    e = random.random()
    x = 0
    y = 0

    if e < p_random_move:
        x, y = random_direction()
    else:
        dict_data = get_json_data(name,url)
        if dict_data == None or len(dict_data)==0:
            return
        policy = analyse(dict_data)
        if policy=="hunting":
            pass
        elif policy == "running away":
            x,y = run_away(dict_data)
            #print("running")
        elif policy == "eat food":
            #print("eating food")
            x,y = find_nearst_food(dict_data)
        elif policy == "split running":
            #print("split running")
            x, y = run_away(dict_data)
            split(driver)

    dangerous_cells.clear()
    food_cells.clear()
    can = driver.find_element_by_tag_name('canvas')
    action = ActionChains(driver)
    action.move_to_element_with_offset(can, x, y)
    action.click()
    action.perform()
    action.reset_actions()




def run_away(dict_data):
    self_info = dict_data.get("visibleCells")
    self_x = self_info[0].get("x")
    self_y = self_info[0].get("y")
    x, y = find_nearst_food2(dict_data)
    #did not find safety foods
    if x==10000:
        dire_dict = {0: (200, 300), 1: (300, 300), 2: (400, 300), 3: (400, 400),
                     4: (400, 500), 5: (300, 500), 6: (200, 500), 7: (200, 400)}
        available_direct = []
        for i in range(8):
            dir_x, dir_y = dire_dict.get(i)
            absolute_x = dir_x-287+self_x
            absolute_y = dir_y - 356 + self_y
            if not is_in_dangerous_area(absolute_x, absolute_y):
                available_direct.append((dir_x, dir_y))
        if len(available_direct) > 0:
            result_x,result_y = np.random.choice(available_direct,1)
            return result_x,result_y
        else:
            index = random.randint(0,7)
            result_x, result_y = dire_dict.get(index)
            return result_x, result_y
    else:
        return x,y

def find_nearst_food2(dict_data):

    self_info = dict_data.get("visibleCells")
    self_x = self_info[0].get("x")
    self_y = self_info[0].get("y")
    self_cells = self_info[0].get("cells")
    self_radius = self_cells[0].get("radius")
    min_x = 10000
    min_y = 10000
    min_distance = 1000000

    if len(food_cells) > 0:
        count = 0
        for food_cell in food_cells:
            x = food_cell.get("x")
            y = food_cell.get("y")

            distance = math.fabs(self_x - x) + math.fabs(self_y - y)
            if distance < min_distance and (not is_in_dangerous_area(x, y)):
                min_x = x
                min_y = y
                min_distance = distance
            count = count + 1
            if count>20:
                break
        if min_x == 10000:
            move_center_direction(self_x,self_y)
        else:
            new_x, new_y = transform(min_x, min_y, self_x, self_y, self_radius)
            return new_x, new_y

    foods = dict_data.get("visibleFood")
    #print("foods num", len(foods))
    if len(foods) > 0:
        for food_dict in foods:
            x = food_dict.get("x")
            y = food_dict.get("y")

            distance = math.fabs(self_x - x) + math.fabs(self_y - y)

            if distance < min_distance and (not is_in_dangerous_area(x, y)):
                min_x = x
                min_y = y
                min_distance = distance
        #print("foods x ,y", min_x, min_y)
        new_x, new_y = transform(min_x, min_y, self_x, self_y, self_radius)
        return new_x, new_y

def find_nearst_food(dict_data):

    self_info = dict_data.get("visibleCells")
    self_x = self_info[0].get("x")
    self_y = self_info[0].get("y")
    self_cells = self_info[0].get("cells")
    self_radius = self_cells[0].get("radius")

    min_x = 10000
    min_y = 10000
    min_distance = 1000000

    if len(food_cells) > 0:
        count = 0
        for food_cell in food_cells:
            x = food_cell.get("x")
            y = food_cell.get("y")

            distance = math.fabs(self_x - x) + math.fabs(self_y - y)
            if distance < min_distance and (not is_in_dangerous_area(x,y)):
                min_x = x
                min_y = y
                min_distance = distance
            count = count+1
            if(count>20):
                break
        new_x, new_y = transform(min_x, min_y, self_x, self_y,self_radius)
        return new_x, new_y

    foods = dict_data.get("visibleFood")
    #print("foods num",len(foods))
    if len(foods) > 0:
        for food_dict in foods:
            x = food_dict.get("x")
            y = food_dict.get("y")

            distance = math.fabs(self_x-x) + math.fabs(self_y-y)
            if distance < min_distance and (not is_in_dangerous_area(x,y)):
                min_x = x
                min_y = y
                min_distance = distance
        #print("foods x ,y", min_x, min_y)
        if min_distance < 800 :
            new_x, new_y = transform(min_x, min_y, self_x, self_y, self_radius)
            return new_x, new_y
    if random.random()< 0.2:
        new_x,new_y = random_direction()
        return new_x,new_y
    else:
        new_x, new_y = move_center_direction(self_x,self_y)
        return new_x,new_y


def is_in_dangerous_area(x,y,dangerous_distance = 350):
    is_dangerous = False
    count = 0
    for danger_cell in dangerous_cells:
        count = count+1
        cell_x = danger_cell.get("x")
        cell_y = danger_cell.get("y")
        if distance(x, y,cell_x , cell_y) < dangerous_distance:
            is_dangerous = True
        if count > 20:
            break
    return is_dangerous



def distance(x1,y1,x2,y2):
    return math.fabs(x1 - x2) + math.fabs(y1 - y2)

def transform(x,y,play_x,play_y,self_radius):

    # d = math.sqrt((x-play_x)**2+(y-play_y)**2)
    # cos = (x-play_x)/d
    # sin = (y-play_y)/d
    new_x = x - play_x + 287
    new_y = y - play_y + 356
    # if x - play_x <0:
    #     new_x = math.fabs(x - play_x + 300 )
    # else:
    #     new_x = math.fabs(x - play_x + 300 )
    # if y - play_y <0:
    #     new_y = math.fabs(y - play_y + 400 )
    # else:
    #     new_y = math.fabs(y - play_y + 400)
    return new_x,new_y

def move_center_direction(play_x,play_y):
    d = math.sqrt((play_x - 1000) ** 2 + (play_y - 1000) ** 2)
    cos = (1000 - play_x) / d
    sin = (1000 - play_y) / d

    new_x =  287 + cos * 200
    new_y =  356 + sin * 200
    return new_x,new_y

def analyse(dict_data,dangerous_distance = 350,emergency_distance = 260,food_distance = 450):
    policy = "eat food"
    self_info = dict_data.get("visibleCells")
    emerny = []

    for i in range(0,len(self_info)):
        name = self_info[i].get("name")
        if name == None:
            self_x = self_info[i].get("x")
            self_y = self_info[i].get("y")
            self_cells = self_info[i].get("cells")
            self_radius = self_cells[0].get("radius")
        else:
            emerny.append(self_info[i])

    for i in range(0,len(emerny)):
        cell_list = emerny[i].get("cells")
        cell_dcit = cell_list[0]
        weight = cell_dcit.get("radius")
        x = cell_dcit.get("x")
        y = cell_dcit.get("y")
        distance = math.fabs(self_x-x) + math.fabs(self_y-y)
        #print("current distance is",distance)
        #print("self radius: ",self_radius)
        #print("weight: ", weight)
        empty_distance = distance - self_radius -weight
        #print("weight distance: ",empty_distance)
        if weight > self_radius*1.1 and empty_distance < dangerous_distance and empty_distance > empty_distance:
            dangerous_cells.append(cell_dcit)
            policy = "running away"
        elif weight > self_radius*1.1 and  distance < emergency_distance:
            policy = "split running"
        elif weight < self_radius*1.1 and distance < food_distance:
            food_cells.append(cell_dcit)
    return policy

def split(driver):
    #print("click split button1.")
    driver.find_element_by_css_selector("#split").click()
    #print("click split button2.")



def get_json_data(name,url="http://localhost:3000/get_data/"):
    url = url + name
    response = requests.get(url)
    if response != None:
        json_data = response.json()
        #print("json_data", json_data)
        return json_data
    else:
        return None



numworkers = multiprocessing.cpu_count()
#print("cpu count: ",numworkers)
numworkers = 10
for i in range(numworkers):
    t = threading.Thread(target=work,args=(str(i)))
    t.start()
