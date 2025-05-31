from playwright.sync_api import sync_playwright
import time
import random
import requests
import os
import hashlib
from PIL import Image
import imageio
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import imagehash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import pickle
#####################
# 1.参数设定
#####################
# 1.1 bilibili爬虫相关
"""
    webp转成png
    参数:
        target_charas (str[]): 爬取目标
        pages (int): 爬取的页数，一页大概500 - 1000张webp
        save_address (str): 保存位置
        now_cookies (dictionary): 可用cookies，记得更新
"""
target_charas = [
    "冰川日菜", "丸山彩", "白鹭千圣", "若宫伊芙", "大和麻弥" # Pastel*Palette
    #"户山香澄", "花园多惠", "牛込里美", "山吹沙绫", "市谷有咲",  # Poppin'Party
    #"美竹兰", "青叶摩卡", "上原绯玛丽", "宇田川巴", "羽泽鸫",  # Afterglow
    #"凑友希那", "冰川纱夜", "今井莉莎", "宇田川亚子", "白金磷子",  # Roselia
    #"弦卷心", "濑田薰", "北泽育美", "松原花音", "奥泽美咲",  # Hello, Happy World!
    #"layer和奏瑞依", "lock朝日六花", "MASKING佐藤益木", "PAREO鳰原令王那", "CHU²珠手知由",  # RAISE A SUILEN
    #"八潮瑠唯", "广町七深", "二叶筑紫", "桐谷透子",  # Morfonica
    #"高松灯", "千早爱音", "要乐奈", "椎名立希", "长崎爽世", # MyGO!!!!!
    #"丰川祥子", "喵梦", "若叶睦", "三角初华", "八幡海铃"  # Ave Mujica
]#
pages = 1
# 保存位置
save_address=f"C:\\Users\\DX110\\Desktop\\pictures_of_bangdream_charactor"
# 可用cookies
now_cookies = [
    {
        "name": "_uuid",
        "value": "B4ADE186-1F84-11013-9E3D-87673F5D8F10D27922infoc",
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1773844727,  # 2026-03-18 时间戳
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax"
    },
    {
        "name": "b_nut",
        "value": "1742299130",
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1777002530,  # 2026-04-22 时间戳
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax"
    },
    {
        "name": "bili_jct",
        "value": "43ee2706761bc2d1e8dcf9444c844d8a",  # 已更新为最新值
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1758030022,  # 2025-11-20 时间戳
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax"
    },
    {
        "name": "bili_ticket",
        "value": "eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDgzMzIwMTIsImlhdCI6MTc0ODA3Mjc1MiwicGx0IjotMX0.mjKcJMXyPP041YMBN3v8ffa8ns_2BxI5gMbNNB0MTfc",  # 已更新为最新值
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1748331952,  # 2025-05-27 时间戳（对应 bili_ticket_expires 字段值）
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax"
    },
    {
        "name": "bp_t_offset_3546872872438586",
        "value": "1070453661462691840",  # 已更新为最新值
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1759441241,  # 2025-06-23 时间戳
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax"
    },
    {
        "name": "buvid3",
        "value": "723332F9-6988-7856-6BEB-169BF843085C33332infoc",
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1777002530,  # 2026-04-22 时间戳
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax"
    },
    {
        "name": "CURRENT_FNVAL",
        "value": "2000",
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1773942035,  # 2026-03-23 时间戳
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax"
    },
    {
        "name": "DedeUserID",
        "value": "3546872872438586",
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1758030022,  # 2025-11-20 时间戳
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax"
    },
    {
        "name": "DedeUserID__ckMd5",
        "value": "f5166823f8de3e44",
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1758030022,  # 2025-11-20 时间戳
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax"
    },
    {
        "name": "LIVE_BUVID",
        "value": "AUTO4817421754878585",
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1776826687,  # 2026-04-21 时间戳
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax"
    },
    {
        "name": "SESSDATA",
        "value": "8dd0db5e%2C1763624815%2Cce1ab%2A52CjBaIrMNWU9AVqpRZN3xdZT54f7FbKRdlO7nsk_-sJdqMGFyIBesLA39FWUKpSBttG4SVmcyZUxmaEIxdmQ1NUFsdXY4UE1XblRHUGY4RG4wQ3hSZWxDSU1WOFp0WF80dXFEOTItOU4yeEM1OUR4SzUtOXhOcWdZd3NXNnl1cEFVLXNCTHllY1BBIIEC",  # 已更新为最新值
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1763624815,  # 注意：根据 SESSDATA 字段的 expires 值（2025-11-20 可能已过期，需确认最新有效期）
        "httpOnly": True,
        "secure": True,
        "sameSite": "Strict"
    },
    {
        "name": "buvid_fp",
        "value": "f5b7b30a93d5d06df74088bcf76bfc5c",
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1777002528,  # 2026-04-22 时间戳
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax"
    },
    {
        "name": "browser_resolution",
        "value": "1912-924",  # 已更新为最新分辨率
        "domain": ".bilibili.com",
        "path": "/",
        "expires": 1773942053,  # 2026-05-24 时间戳（注意：当前时间为 2025-05-24，该过期时间在未来）
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax"
    },
]

# 1.2 格式转换相关
"""
    webp转成png
    参数:
        target_file (str): 要转换的文件夹
"""
target_file = r'C:\Users\DX110\Desktop\pictures_of_bangdream_charactor'

# 1.3 删除重复相关
"""
    安全地遍历根目录及其子目录，删除重复图片
    参数:
        root_dir (str): 根目录路径
        dry_run (bool): 是否只显示结果不执行删除
        interactive (bool): 是否交互式确认删除
        min_size (int): 最小文件大小(KB)，小于此大小的文件会被忽略
        threshold (int): 汉明距离阈值，0相当于只用MD5，越大越宽松
"""
r_root_dir = r'C:\Users\DX110\Desktop\pictures_of_bangdream_charactor'
r_dry_run = False
r_interactive = False
r_min_size = 5
r_threshold = 5

# 1.4 删除过小图片相关
"""
    删除指定目录及其子目录中所有小于指定大小的文件
    参数:
        target_dir (str): 要检查的根目录
        size_threshold (int): 文件大小阈值（字节），默认为15KB
        interact (bool): 开启删除确认
"""
d_target_dir = r'C:\Users\DX110\Desktop\pictures_of_bangdream_charactor'
d_size_threshold = 15
d_interact = False

# 1.5 降噪相关
"""
    批量对目录下所有图片进行降噪处理
    参数：
        root_dir (str): 图片根目录
        methods (list): 降噪方法列表，按顺序执行
                        可选 "gaussian"（高斯滤波）、"threshold"（阈值法）、"kmeans"（聚类法）
        kernel_size (pair<int, int>): 高斯滤波参数
        sigma (int): 高斯滤波参数
        threshold (int): 阈值法参数
        n_clisters (int): K-means法参数
        max_iter (int): K-means法参数
"""
b_root_dir=b_root_dir = r'C:\Users\DX110\Desktop\pictures_of_bangdream_charactor'
b_methods = ["threshold", "gaussian", "kmeans"]
d_kernel_size=(3, 3),
d_sigma=1.0,
d_threshold=127,
d_n_clusters=4,
d_max_iter=100

# 1.6 预处理暴力降噪发相关
"""
    一种奇怪的的算法
    参数:
        input_size: 模型图片尺寸
        root_dir: 源图片目录
        model_path: .h5模型文件路径c
        output_dir: 符合条件的图片保存目录，默认为None（直接在原目录删除不符合的图片）
        threshold: 预测置信度阈值
        target_size: 模型输出尺寸
        class_names: 类别名称列表
        dry_run: 是否只预览不执行实际操作
"""
o_input_size = (224, 224)
o_root_dir = r'C:\Users\DX110\Desktop\pictures_of_bangdream_charactor'
o_model_path = r"C:\Users\DX110\Desktop\model_of_bangdream_charactor_identify\anime_char_model.h5"
o_preprocessing_params_path = r"C:\Users\DX110\Desktop\model_of_bangdream_charactor_identify\preprocessing_params.pkl"
o_train_dir = r'C:\Users\DX110\Desktop\pictures_of_bangdream_charactor'
o_output_dir = None
o_threshold = 0
o_target_size = o_input_size
o_dry_run = False
#####################
# 2.bilibili爬虫
#####################
'''
    因为代码根据bilibili图文栏定制，所以无法用于其它网页
    用了两三个月一直好使
    慎改参数，小心IP被封禁
    记得更新cookies
'''
# 获取url
def scrape_bilibili(chara, pages):
    url_list=[]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7"
        )
        cookies = now_cookies
        context.add_cookies(cookies)
        page = context.new_page()
        for page_num in range(1, 1 + pages):
            url = f"https://search.bilibili.com/article?keyword={chara}&page={page_num}"
            page.goto(url, wait_until='networkidle')
            try:
                page.wait_for_selector('.b-article-card', timeout=20000, state='visible') 
                articles = page.query_selector_all('.b-article-card')
                for article in articles:
                    title_elem = article.query_selector('.text1')
                    if title_elem:
                        title = title_elem.text_content().strip()
                        href = title_elem.get_attribute('href')
                        print(f"标题：{title}\n链接：{href}\n")
                        if href:
                            url_list.append(href)
            except Exception as e:
                print(f"处理页面 {page_num} 时出现错误: {e}")
            time.sleep(random.uniform(2, 4))
        browser.close()
    return url_list
# 保存图片
def scrape_picture(urls, chara):
    save_dir = os.path.join(save_address, chara)
    os.makedirs(save_dir, exist_ok = True)
    total_urls = len(urls)
    for index, url in enumerate(urls, start=1):
        if url.startswith('//'):
            url = 'https:' + url
        with sync_playwright() as q:
            browser = q.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                locale="zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7"
            )
            cookies = now_cookies
            context.add_cookies(cookies)
            page = context.new_page()
            try:
                page.goto(url, wait_until='networkidle', timeout=20000)
                pictures = page.query_selector_all('div.b-img picture.b-img__inner img')
                for picture in pictures:
                    img_src = picture.get_attribute('src')
                    if img_src and img_src.startswith('//'):
                        img_src = 'https:' + img_src
                        img_name = img_src.split('/')[-1]
                        img_path = os.path.join(save_dir, img_name)
                        try:
                            response = requests.get(img_src)
                            if response.status_code == 200:
                                with open(img_path, 'wb') as f:
                                    f.write(response.content)
                                print(f"成功保存图片：{img_path}")
                            else:
                                print(f"下载图片失败，状态码：{response.status_code}")
                        except Exception as e:
                            print(f"下载图片时出现错误: {e}")
            except Exception as e:
                print(f"访问页面 {url} 时出现错误: {e}")
            finally:
                browser.close()
        percentage = (index / total_urls) * 100
        print(f"已完成: {percentage:.2f}%")
# 启动爬虫
def scrape_begin():
    for chara in target_charas:
        urls=scrape_bilibili(chara, pages)
        scrape_picture(urls, chara)

#####################
# 3.将爬取的webp文件转为png可用于训练的格式
#####################
'''
    没有日志
    这个很安全，如果想保留webp格式文件直接去备份
'''

# 转换核心函数
def convert_webp_to_png(src_dir):
    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            if filename.lower().endswith(".webp"):
                src_path = os.path.join(root, filename)
                dest_path = os.path.join(root, os.path.splitext(filename)[0] + ".png")
                try:
                    img = imageio.imread(src_path)
                    imageio.imwrite(dest_path, img)
                    print(f"已转换：{src_path} -> {dest_path}")
                    os.remove(src_path)
                except Exception as e:
                    print(f"处理失败：{src_path}，错误：{e}")
# 转换启动函数
def convert_begin():
    convert_webp_to_png(target_file)

#####################
# 4.图片去重
#####################
'''
    !!!!警告!!!!
    该功能测试中，可能导致严重的误删，建议启动交互
'''

# 计算MD5哈希值，已经弃用，慎重启用，有严重的哈希冲突
def get_image_hash(file_path, block_size=65536):
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(block_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
       print(f"无法计算MD5哈希值: {file_path}，由于错误: {e}")
       return None
# 计算感知哈希.
def get_image_phash(file_path, hash_size=8):
    try:
        with Image.open(file_path) as img:
            return imagehash.phash(img, hash_size=hash_size)
    except Exception as e:
        print(f"无法计算感知哈希值: {file_path}，由于错误: {e}")
        return None
# 获取大小和尺寸信息
def get_image_info(file_path):
    try:
        with Image.open(file_path) as img:
            return img.size, img.format
    except Exception:
        return None, None
# 根据上述几个信息开始删除，核心
def remove_similar_images(root_dir, threshold=5, dry_run=True, interactive=True, min_size=1024):
    hash_dict = {}
    total_files = 0
    duplicates = 0
    ignored = 0
    
    print(f"开始扫描目录: {root_dir}")
    start_time = time.time()

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
            try:
                file_size = os.path.getsize(file_path) / 1024  # KB
                if file_size < min_size:
                    ignored += 1
                    continue
            except OSError:
                continue
                
            total_files += 1
            img_hash = get_image_phash(file_path)
            
            if img_hash is None:
                continue
                
            is_duplicate = False
            for stored_hash in hash_dict:
                if img_hash - stored_hash <= threshold:
                    hash_dict[stored_hash].append(file_path)
                    duplicates += 1
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                hash_dict[img_hash] = [file_path]
    
    # 筛选出真正的相似文件组
    similar_groups = {h: paths for h, paths in hash_dict.items() if len(paths) > 1}
    
    duration = time.time() - start_time
    print(f"扫描完成，耗时: {duration:.2f}秒")
    print(f"共扫描 {total_files} 个图片文件，找到 {len(similar_groups)} 组相似图片，包含 {duplicates} 个相似项")
    print(f"忽略了 {ignored} 个小于 {min_size}KB 的文件")
    
    if not similar_groups:
        print("没有找到相似图片")
        return
    
    deleted_count = 0
    delete_all = False
    
    for img_hash, file_paths in similar_groups.items():
        info_list = []
        for path in file_paths:
            size, format = get_image_info(path)
            info_list.append((os.path.getsize(path), size, format))
        
        keep_index = 0
        for i, info in enumerate(info_list):
            if info[0] > info_list[keep_index][0]:
                keep_index = i
        
        print("\n发现相似图片组:")
        for i, (path, info) in enumerate(zip(file_paths, info_list)):
            size_mb = info[0] / (1024 * 1024)
            status = "保留" if i == keep_index else "删除"
            print(f"  [{status}] {os.path.basename(path)} - {size_mb:.2f}MB, {info[1]}, {info[2]}")
        
        if interactive:
            if delete_all:
                print("自动删除中")
                response = 'y'
            else:
                response = input("确认删除这些相似图片吗？(y/n/all): ").lower()
            
            if response == 'n':
                print("已跳过此组")
                continue
            elif response == 'all':
                delete_all = True
                print("已启用自动确认，后续所有相似图片组将被直接删除")
            elif response != 'y':
                print("已跳过此组")
                continue
        
        for i, path in enumerate(file_paths):
            if i != keep_index:
                if not dry_run:
                    try:
                        os.remove(path)
                        deleted_count += 1
                        print(f"已删除: {path}")
                    except Exception as e:
                        print(f"删除失败: {path}, 错误: {e}")
                else:
                    print(f"[模拟] 已删除: {path}")
    
    print(f"\n操作完成! 共删除 {deleted_count} 个相似图片")
# 删除启动函数
def remove_duplicate_begin():
    remove_similar_images(r_root_dir, threshold=r_threshold, dry_run=r_dry_run, interactive=r_interactive, min_size=r_min_size)

#####################
# 5.删除过小文件
#####################
'''
    !!!!警告!!!!
    这个没有逐张交互确认
'''

# 删除核心函数
def delete_small_files(root_dir, size_threshold=15 * 1024, interact = True):
    
    if not os.path.isdir(root_dir):
        print(f"错误: 目录 '{root_dir}' 不存在")
        return

    files_to_delete = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                file_size = os.path.getsize(file_path)
                if file_size < size_threshold:
                    files_to_delete.append((file_path, file_size))
            except OSError as e:
                print(f"无法获取文件 {file_path} 的大小: {e}")
    
    if not files_to_delete:
        print("没有找到小于15KB的文件")
        return
    
    print(f"找到 {len(files_to_delete)} 个小于 {size_threshold/1024}KB 的文件:")
    for file_path, size in files_to_delete[:10]:
        print(f"  {file_path} ({size/1024:.2f}KB)")
    if len(files_to_delete) > 10:
        print(f"  ... 和其他 {len(files_to_delete)-10} 个文件")
    
    while True:
        if interact:
            response = 'y'
        else:
            response = input("确定要删除这些文件吗？(y/n): ").strip().lower()
        if response == 'y':
            deleted_count = 0
            for file_path, _ in files_to_delete:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except OSError as e:
                    print(f"无法删除文件 {file_path}: {e}")
            print(f"成功删除 {deleted_count} 个文件")
            break
        elif response == 'n':
            print("操作已取消")
            break
        else:
            print("请输入 'y' 或 'n'")
# 删除启动函数
def delete_small_begin():
    print(f"正在检查目录: {d_target_dir}")
    delete_small_files(d_target_dir, d_size_threshold * 1024, interact = d_interact)

#####################
# 6. 降噪处理
#####################
"""
    没有日志，AI写的，基本没用过
"""
# 高斯滤波
def gaussian_filter(img_path, kernel_size=(5, 5), sigma=0):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            return False
        filtered_img = cv2.GaussianBlur(img, kernel_size, sigma)
        cv2.imwrite(img_path, filtered_img)
        print(f"高斯滤波完成：{img_path}")
        return True
    except Exception as e:
        print(f"高斯滤波失败：{img_path}，错误：{e}")
        return False
# 阈值法
def threshold_denoise(img_path, threshold=100, max_val=255, threshold_type=cv2.THRESH_BINARY):
    try:
        img = cv2.imread(img_path, 0)
        _, thresh = cv2.threshold(img, threshold, max_val, threshold_type)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print(f"警告：{img_path} 中未检测到轮廓")
            return False
        max_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
        result = cv2.bitwise_and(img, mask)
        cv2.imwrite(img_path, result)
        print(f"阈值降噪完成：{img_path}")
        return True
    except Exception as e:
        print(f"阈值降噪失败：{img_path}，错误：{e}")
        return False
# K-means 聚类降噪（适用于颜色噪声）
def kmeans_denoise(img_path, n_clusters=3, max_iter=100):
    try:
        img = Image.open(img_path)
        pixels = np.array(img).reshape(-1, 3)
        pixels = np.float32(pixels)
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42).fit(pixels)
        labels = kmeans.labels_
        centers = np.uint8(kmeans.cluster_centers_)
        segmented_pixels = centers[labels.flatten()].reshape(img.size[1], img.size[0], 3)
        cv2.imwrite(img_path, cv2.cvtColor(segmented_pixels, cv2.COLOR_BGR2RGB))
        print(f"K-means 降噪完成：{img_path}")
        return True
    except Exception as e:
        print(f"K-means 降噪失败：{img_path}，错误：{e}")
        return False
# 批量降噪处理
def batch_denoise(root_dir, methods=[], **kwargs):
    
    if not methods:
        print("警告：未指定任何降噪方法")
        return
    
    valid_methods = {"gaussian", "threshold", "kmeans"}
    for method in methods:
        if method not in valid_methods:
            print(f"错误：不支持的降噪方法 {method}，可选 {valid_methods}")
            return
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(dirpath, filename)
                temp_path = img_path + ".temp"
                
                current_path = img_path
                
                for method in methods:
                    if method == "gaussian":
                        success = gaussian_filter(
                            current_path, 
                            kernel_size=kwargs.get("kernel_size", (5, 5)),
                            sigma=kwargs.get("sigma", 0),
                        )
                    elif method == "threshold":
                        success = threshold_denoise(
                            current_path, 
                            threshold=kwargs.get("threshold", 100),
                            threshold_type=kwargs.get("threshold_type", cv2.THRESH_BINARY),
                        )
                    elif method == "kmeans":
                        success = kmeans_denoise(
                            current_path, 
                            n_clusters=kwargs.get("n_clusters", 3),
                            max_iter=kwargs.get("max_iter", 100),
                        )
                    
                    if success:
                        if os.path.exists(temp_path):
                            if current_path != img_path:
                                os.remove(current_path)
                            current_path = temp_path
                    else:
                        print(f"警告：{method} 处理失败，跳过后续处理")
                        break
                if current_path != img_path and os.path.exists(current_path):
                    os.replace(current_path, img_path)
                    print(f"多重降噪完成：{img_path}")
# 降噪启动函数
def denoise_begin():
    """详细参数建议问AI"""
    batch_denoise(
        root_dir=b_root_dir,
        methods = b_methods,
        kernel_size=d_kernel_size,
        sigma=d_sigma,
        threshold=d_threshold,
        n_clusters=d_n_clusters,
        max_iter=d_max_iter
    )

#####################
# 7. 预处理暴力预测降噪法
#####################
"""
    突发奇想的算法
    对于五维向量，进行清洗测试，数据如下

    训练一个识别bangdream企划中乐队pastel*palette五名成员 丸山彩 冰川日菜 白鹭千圣 大和麻弥 若宫伊芙 的计算机视觉模型

    说明：第二次训练图片来源于人工筛选和第一次清洗结果。由于每次爬虫都是爬取相同网页，所以后续清洗会出现某些训练图片数量数据增长很多而某些几乎无增长

    组名    首次训练图片    首次模型测试数量    首次模型测试准确数量    首次测试模型准确度    首次清洗结果数量    首次清洗准确度    第二次训练图片    第二次清洗结果数量    第二次清洗准确数量    第二次清洗准确率    第二次模型测试数量    第二次模型测试准确数量    第二次模型测试准确度
   丸山彩      198                74                   29                39.19%             统计失败*         统计失败           395                31                   21                67.74%               74                      7                   9.46%
  冰川日菜     269               256                  112                43.75%                                                 339               102                   98                96.08%              256                    197                  76.95%
  白鹭千圣     487               463                  337                72.79%                                                 531               148                  148               100.00%              463                    155                  33.48%
  大和麻弥     367               353                  102                28.90%                                                 416                99                   85                85.86%              353                     86                  24.36%
  若宫伊芙     382               172                   52                30.23%                                                 488               278*                 199                71.58%              172                     86                  50.00%
                                                                                        *注：忘记统计                                            *注：这是白毛，导致一堆类似角色因为图片颜色/构图/灯光/背景等问题无法识别。人眼也难识别。
    数据说明：首次训练图片手动降噪。第0组为少数据训练组，第1组为特征明显组，第2组为大量数据组，第3组为对照组，第4组为特征不明显组。第二次训练图片为第一次加第一次筛选图片合并去重。
    对于特征明显的实验组，运行该方法进行训练数据的获取与清洗再次训练可以提升模型准确度。
    对于已经接近过拟合的实验组，该方法将会变成一个高效准确的爬虫
    <notice>: 进一步实验将把首次训练图片完全采用爬来并初次降噪的数据拿来训练

    数据分析：

    特征明显组（冰川日菜）的有效性
    对比第1组的两次模型准确度数据
    推测原因：人工筛选与首次清洗结果合并后，有效样本占比增加，模型能更精准捕捉关键特征。

    过拟合组（白鹭千圣）的爬虫化现象
    对比第2组和第3组
    首次模型准确度72.79%（可能已经过拟合），第二次清洗准确率100%，模型测试准确度暴跌至33.48%。
    推测原因：过拟合模型对训练数据过度依赖，清洗后新增数据与原模型特征空间不匹配，导致泛化能力下降，清洗过程仅成为数据爬取工具，未提升模型质量。

    少数据组（丸山彩）的异常表现
    第二次清洗准确率67.74%（较低数据），模型测试准确度从39.19% 降至9.46%。
    推测原因：首次训练数据手动降噪后特征纯净，但有效特征不足，二次训练合并时引入大量噪声。

    特征不明显组（若宫伊芙）的挑战
    标注问题：因 “白毛特征” 导致人难以识别，清洗准确率71.58%，反映特征模糊场景需依赖更精细的标注规则（如颜色阈值、构图模板）。
    模型提升有限：测试准确度从30.23% 提升至50%，表明即使数据量增加，特征模糊仍制约模型上限。

    对照组（大和麻弥）的基准价值
    数据对比：首次/二次训练准确度分别为28.90%/24.36%，与少数据组（丸山彩）相比，显示初始数据量差异可能导致过拟合风险（丸山彩数据少但首次准确度更高）。


    下一步实验计划
    验证假设：使用完全未手动降噪的爬取数据进行首次训练（原计划），对比 “纯爬虫数据 + 清洗” 与 “手动降噪数据” 的模型表现，明确数据预处理的必要性。
    
    AI提出的技术优化：
    对特征模糊组（如若宫伊芙）增加多模态特征提取（如 OCR 文本标签、视频帧动态特征）。
    针对过拟合风险，引入数据增强算法（如旋转、亮度变换）扩展训练集多样性。
    标注体系升级：建立 “特征权重评分表”，对颜色、构图、背景等维度量化打分，减少人工标注主观误差。

    核心启示：1.数据清洗的价值高度依赖特征可分性与初始数据质量，后续实验需强化特征工程与自动化标注规则，降低人工干预成本。
             2.模型质量对该方法可能有重要影响，推测为第二组特征明显导致模型拟合能力强。
    
    应用方向：1.可以多次训练，来爬取特征异常明显的图片。
             2.数据表明该方法对于模型预测能力提升有积极影响。
"""

# 算法类
class Ose_preprocessing_traversal:
    def __init__(self, model_path, preprocessing_params_path, train_dir, target_size=(224, 224)):
        self.model_path = model_path
        self.train_dir = train_dir
        self.model = load_model(model_path)
        self.target_size = target_size
        
        self.class_indices = self._get_class_indices()
        
        with open(preprocessing_params_path, 'rb') as f:
            preprocessing_params = pickle.load(f)
        self.mean = preprocessing_params['mean']
        self.std = preprocessing_params['std']
    
    def _get_class_indices(self):
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        if not os.path.isdir(self.train_dir):
            raise ValueError(f"训练数据目录不存在: {self.train_dir}")
        if not any(os.path.isdir(os.path.join(self.train_dir, d)) for d in os.listdir(self.train_dir)):
            raise ValueError(f"训练数据目录 {self.train_dir} 中没有类别子文件夹")
        
        temp_generator = datagen.flow_from_directory(
            self.train_dir,
            target_size=self.target_size,
            batch_size=32,
            class_mode='categorical',
            subset='training',
            shuffle=False
        )
        
        class_indices = temp_generator.class_indices
        print(f"成功获取类别索引映射: {class_indices}")
        return class_indices

    def predict(self, img_path):
        try:
            img = image.load_img(img_path, target_size=self.target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0
            
            if self.mean is not None and self.std is not None:
                x = (x - self.mean) / (self.std + 1e-7)
            
            predictions = self.model.predict(x, verbose=0)
            class_index = np.argmax(predictions[0])
            confidence = predictions[0][class_index] * 100
            class_names = list(self.class_indices.keys())
            predicted_class = class_names[class_index]
            
            print(f"\n图像: {os.path.basename(img_path)} 的预测分布:")
            for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
                print(f"{i+1}. {class_name}: {prob*100:.2f}%")
            
            return predicted_class, confidence
        except Exception as e:
            print(f"错误：处理 {img_path} 时出错: {e}")
            return None, 0

    def is_match_folder_name(self, img_path, folder_name, threshold=0.5):
        predicted_class, confidence = self.predict(img_path)
        if predicted_class is None:
            return False
        
        print(f"预测类别: {predicted_class}, 文件夹名称: {folder_name}, 置信度: {confidence:.2f}%")
        return predicted_class.lower() == folder_name.lower() and confidence >= threshold * 100
# 核心函数
def filter_images_by_prediction(root_dir, model_path, train_dir, preprocessing_params_path, output_dir=None, threshold=0.5, target_size=(224, 224), dry_run=True):
    classifier = Ose_preprocessing_traversal(model_path, preprocessing_params_path, train_dir, target_size)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    total_images = 0
    kept_images = 0
    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        print(f"\n正在处理: {class_folder}")
        if output_dir:
            output_class_path = os.path.join(output_dir, class_folder)
            if not os.path.exists(output_class_path):
                os.makedirs(output_class_path)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            total_images += 1
            if classifier.is_match_folder_name(img_path, class_folder, threshold):
                kept_images += 1
                print(f"✅ 保留: {img_name} (预测为 {class_folder})")
                if output_dir and not dry_run:
                    shutil.copy2(img_path, os.path.join(output_class_path, img_name))
            else:
                print(f"❌ 删除: {img_name} (未预测为 {class_folder})")
                if not output_dir and not dry_run:
                    os.remove(img_path)
    
    print(f"\n总结:")
    print(f"共处理图像: {total_images}")
    print(f"保留: {kept_images} ({(kept_images/total_images)*100:.2f}%)")
    print(f"删除: {total_images - kept_images} ({((total_images-kept_images)/total_images)*100:.2f}%)")
# 预处理遍历启动函数
def Ose_preprocessing_traverse_begin():
    filter_images_by_prediction(
        root_dir=o_root_dir,
        model_path=o_model_path,
        train_dir=o_train_dir,
        preprocessing_params_path=o_preprocessing_params_path,
        output_dir=o_output_dir,
        threshold=o_threshold,
        target_size=o_input_size,
        dry_run=o_dry_run
    )
#####################
# 8. 一键启动
#####################
def auto_execute(customer_needs):
    if customer_needs[0]:
        scrape_begin()
    if customer_needs[1]:
        convert_begin()
    if customer_needs[2]:
        remove_duplicate_begin()
    if customer_needs[3]:
        delete_small_begin()
    if customer_needs[4]:
        denoise_begin()
    if customer_needs[5]:
        Ose_preprocessing_traverse_begin()

##############################################################################################################################################################################################################################
# 在这里调参数开始
# 这几个依次是 爬虫 格式转换 去重 删除小图片 降噪 预处理暴力降噪， True是执行，False是不执行。
customer_needs=[True, True, True, True, True, False]

if __name__ == "__main__":
    auto_execute(customer_needs)