# -*- coding: utf-8 -*-
# @Time       : 2025/9/29 23:39
# @File       : gripper_wifi_controller.py.py
# @Description: 通过wifi控制夹爪
# -*- coding: utf-8 -*-
# @Time       : 2025/9/29 23:39
# @File       : gripper_wifi_controller.py
# @Description: 通过WiFi控制ESP32夹爪

import platform
import socket
import subprocess
import threading
import time
import requests
import json
import re
import sys
from urllib.parse import urlparse

# HTTP端口
HTTP_PORT = 80
# 超时设置
TIMEOUT = 3
# 要搜索的设备标识符
DEVICE_NAME = "ESP32"
# 最大线程数
MAX_THREADS = 50


class GripperWiFiController:
    def __init__(self):
        self.esp32_ip = None
        self.is_scanning = False
        self.last_command_time = 0

    def scan_network(self):
        """扫描局域网寻找ESP32设备，通过HTTP响应识别"""
        self.is_scanning = True
        print("开始扫描局域网中的ESP32设备...")
        self.esp32_ip = None

        # 获取本地IP和子网
        local_ip = self._get_local_ip()
        if not local_ip:
            print("无法获取本地IP地址")
            self.is_scanning = False
            return None

        # 解析子网
        subnet_parts = local_ip.split(".")[:3]
        subnet = "".join([part + "." for part in subnet_parts])

        # 多线程扫描
        threads = []
        for i in range(1, 255):
            if self.esp32_ip:  # 如果已经找到，就停止扫描
                break

            ip = subnet + str(i)
            thread = threading.Thread(target=self._check_ip_http, args=(ip,))
            threads.append(thread)
            thread.start()

            # 限制并发线程数
            if len(threads) >= MAX_THREADS:
                for t in threads:
                    t.join()
                threads = []

        # 等待剩余线程
        for t in threads:
            t.join()

        self.is_scanning = False

        if self.esp32_ip:
            print(f"找到ESP32夹爪控制器，IP地址: {self.esp32_ip}")
        else:
            print("未找到ESP32设备，请检查连接")

        return self.esp32_ip

    def _get_local_ip(self):
        """获取本地IP地址"""
        try:
            # 创建UDP套接字连接到外部地址来获取本地IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 不实际连接，只是获取本地IP
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception as e:
            print(f"获取本地IP出错: {e}")
            return None

    def _check_ip_http(self, ip):
        """通过HTTP请求检查指定IP是否为ESP32设备"""
        if self.esp32_ip:  # 如果已经找到，就停止扫描
            return

        try:
            # 尝试发送HTTP请求到ESP32的服务器
            url = f"http://{ip}:{HTTP_PORT}/STATUS"
            response = requests.get(url, timeout=TIMEOUT)

            # 检查响应是否来自ESP32夹爪控制器
            if response.status_code == 200:
                content = response.text.strip()
                # 检查响应内容格式是否匹配
                if content.startswith("OK: "):
                    # 尝试解析角度值
                    try:
                        angle = int(content[4:])
                        print(f"发现ESP32夹爪控制器: {ip}, 当前角度: {angle}")
                        self.esp32_ip = ip
                    except ValueError:
                        pass
        except Exception:
            pass

    def _check_ip_ping(self, ip):
        """先ping检查IP是否可达"""
        try:
            # 先ping通IP
            param = "-n 1" if platform.system().lower() == "windows" else "-c 1"
            result = subprocess.call(["ping", param, ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result == 0
        except Exception:
            return False

    def send_command(self, command):
        """发送HTTP控制命令到ESP32"""
        if not self.esp32_ip:
            print("未找到ESP32设备，请先扫描网络")
            return False

        # 限制命令发送频率，避免过于频繁
        current_time = time.time()
        if current_time - self.last_command_time < 0.5:
            print("命令发送过于频繁，请稍等再试")
            return False

        self.last_command_time = current_time

        try:
            url = f"http://{self.esp32_ip}:{HTTP_PORT}/{command}"
            response = requests.get(url, timeout=TIMEOUT)

            if response.status_code == 200:
                content = response.text.strip()
                print(f"命令 '{command}' 发送成功，响应: {content}")
                return True
            else:
                print(f"发送命令失败，HTTP状态码: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"无法连接到设备 {self.esp32_ip}，设备可能已离线")
            # 可能IP已经变化，标记为需要重新扫描
            self.esp32_ip = None
            return False
        except Exception as e:
            print(f"发送命令时发生错误: {e}")
            return False

    def get_gripper_status(self):
        """获取夹爪当前状态"""
        return self.send_command("STATUS")

    def open_gripper(self):
        """打开夹爪"""
        return self.send_command("OPEN")

    def close_gripper(self):
        """关闭夹爪"""
        return self.send_command("CLOSE")


# 主程序
if __name__ == "__main__":
    controller = GripperWiFiController()

    print("====== ESP32夹爪WiFi控制器 ======")
    print("正在尝试自动发现设备...")

    # 扫描网络
    controller.scan_network()

    # 如果没找到，手动输入IP
    if not controller.esp32_ip:
        manual_ip = input("请手动输入ESP32的IP地址 (或按回车退出): ")
        if not manual_ip:
            print("程序退出")
            sys.exit(0)
        controller.esp32_ip = manual_ip

    # 主控制循环
    while True:
        print("\n====== 夹爪控制器 ======")
        print("当前连接的设备IP: {}".format(controller.esp32_ip or "未连接"))
        print("1. 打开夹爪")
        print("2. 关闭夹爪")
        print("3. 查看夹爪状态")
        print("4. 重新扫描网络")
        print("5.退出")

        choice = input("请选择操作 (1-5): ")

        if choice == "1":
            controller.open_gripper()
        elif choice == "2":
            controller.close_gripper()
        elif choice == "3":
            controller.get_gripper_status()
        elif choice == "4":
            controller.scan_network()
        elif choice == "5":
            print("程序退出")
            break
        else:
            print("无效的选择，请重新输入")
