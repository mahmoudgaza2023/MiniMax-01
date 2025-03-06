from collections import defaultdict 
import numpy as np 
import os
import json
import argparse 

def modify_gt(gt):  
    match gt:
        case "1. 钢琴\n2. 小提琴\n3. 吉他":
            gt_list = ["钢琴", "小提琴", "吉他"]
        case "1. 生机勃勃\n2. 春暖花开\n3. 万物复苏":
            gt_list = ["生机勃勃", "春暖花开", "万物复苏"]
        case "体型小巧，羽毛灰褐，\n喜欢在城市中觅食，叽叽喳喳很热闹。":
            gt_list = ["体型小巧", "羽毛灰褐", "喜欢在城市中觅食", "叽叽喳喳很热闹"]
        case "1. 韩信\n2. 岳飞\n3. 霍去病":
            gt_list = ["韩信", "岳飞", "霍去病"]
        case "蔚蓝无垠，波涛汹涌，生命的摇篮。":
            gt_list = ["蔚蓝无垠", "波涛汹涌", "生命的摇篮"]
        case "1. 苹果\n2. 香蕉\n3. 橙子":
            gt_list = ["苹果", "香蕉", "橙子"]
        case "蝉鸣阵阵，知了此起彼伏。\n树荫下，老人们悠闲地下着棋。\n孩童嬉戏，欢笑声传遍公园。":
            gt_list = ["蝉鸣阵阵，知了此起彼伏", "树荫下，老人们悠闲地下着棋", "孩童嬉戏，欢笑声传遍公园"]
        case "1. 微积分\n2. 线性代数\n3. 概率论":
            gt_list = ["微积分", "线性代数", "概率论"]
        case "红艳如火，娇嫩欲滴，\n花瓣层叠，芳香四溢。":
            gt_list = ["红艳如火", "娇嫩欲滴", "花瓣层叠", "芳香四溢"]
        case "在南极的冰山之巅，\n企鹅们舞动着短小的翅膀。\n身披黑白礼服，步伐蹒跚，\n在寒风中，它们笑对严霜。":
            gt_list = ["在南极的冰山之巅", "企鹅们舞动着短小的翅膀", "身披黑白礼服", "步伐蹒跚", "在寒风中", "它们笑对严霜"]
        case "On the peak of the Antarctic iceberg,\nPenguins dance with tiny wings.\nWearing black and white tuxedos, stumbling steps,\nThey smile at the severe frost in the cold wind.":
            gt_list = ["On the peak of the Antarctic iceberg", "Penguins dance with tiny wings", "Wearing black and white tuxedos", "stumbling steps", "They smile at the severe frost in the cold wind"] 
        case "Red as fire, delicate and dripping,\nPetals layered, fragrance overflowing.": 
            gt_list = ["Red as fire", "delicate and dripping", "Petals layered", "fragrance overflowing"] 
        case "1. Calculus\n2. Linear Algebra\n3. Probability Theory": 
            gt_list = ["Calculus", "Linear Algebra", "Probability Theory"] 
        case "Cicadas chirping, the sounds rise and fall.\nUnder the shade, elders leisurely play chess.\nChildren play, laughter fills the park.": 
            gt_list = ["Cicadas chirping, the sounds rise and fall", "Under the shade, elders leisurely play chess", "Children play, laughter fills the park"] 
        case "1. Apple\n2. Banana\n3. Orange": 
            gt_list = ["Apple", "Banana", "Orange"] 
        case "Vast and blue, waves surging, cradle of life.": 
            gt_list = ["Vast and blue", "waves surging", "cradle of life"] 
        case "1. Han Xin\n2. Yue Fei\n3. Huo Qubing":
            gt_list = ["Han Xin", "Yue Fei", "Huo Qubing"] 
        case "Small in size, gray-brown feathers,\nLikes to forage in the city, chirping lively.":
            gt_list = ["Small in size", "gray-brown feathers", "Likes to forage in the city", "chirping lively"] 
        case "1. Piano\n2. Violin\n3. Guitar":
            gt_list = ["Piano", "Violin", "Guitar"] 
        case "1. Vibrant\n2. Fresh\n3. Warm": 
            gt_list = ["Vibrant", "Fresh", "Warm"] 
        case _:
            raise ValueError(f"GT not found: {gt}") 
    return gt_list 


def score_response(response, gt_label, language):  
    if language=='chinese' and ('抱歉' in response or '没有之前的对话' in response):
        return 0 
    if language=='english' and ('sorry' in response.lower() or 'no previous conversation' in response.lower()):
        return 0 
    gt_list = modify_gt(gt_label) 
    score = np.mean([1 if gt in response else 0 for gt in gt_list])  
    return score 

def process_log(response_log): 
    gt_label = response_log['label']
    response = ''
    return gt_label, response 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--language", type=str, default="english")
    args = parser.parse_args() 

    for i, file in enumerate(os.listdir(args.input_dir)):
        token_length_total_score = [] 
        p = os.path.join(args.input_dir, file)
        with open(p, "r") as f:  
            lines = f.readlines()
            response_logs = [json.loads(line) for line in lines]  
        for response_log in response_logs: 
            gt_label, response = process_log(response_log) 
            score = score_response(response, gt_label, args.language)  
            token_length_total_score.append(score) 
        token_length_mean_score = np.mean(token_length_total_score) 
        print(f"File: {file}, Score: {token_length_mean_score}") 
