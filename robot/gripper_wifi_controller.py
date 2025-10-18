# -*- coding: utf-8 -*-
# @Time       : 2025/10/2 10:00
# @File       : gripper_wifi_controller.py
# @Description: 通过WiFi控制ESP32夹爪

import socket
import threading
import time
import sys

import requests

# 配置参数
HTTP_PORT = 80
TIMEOUT = 3
DEVICE_NAME = "ESP32"
MAX_THREADS = 50


class GripperWiFiController:
    def __init__(self):
        self.esp32_ip = None
        self.is_scanning = False
        self.last_command_time = 0

    def connect(self):
        """建立与ESP32夹爪的连接，包括网络扫描和手动输入IP"""
        print("开始连接ESP32夹爪控制器...")

        # 先尝试自动扫描
        found_ip = self.scan_network()
        if found_ip:
            self.esp32_ip = found_ip
            print(f"已成功连接到设备: {self.esp32_ip}")
            return True

        # 自动扫描失败，尝试手动输入
        manual_ip = input("请手动输入ESP32的IP地址 (或按回车取消): ")
        if manual_ip:
            self.esp32_ip = manual_ip
            # 验证手动输入的IP是否有效
            if self._verify_connection():
                print(f"已成功连接到设备: {self.esp32_ip}")
                return True
            else:
                print(f"无法验证IP: {manual_ip} 的有效性")
                self.esp32_ip = None
                return False

        print("连接已取消")
        return False

    def open(self):
        """打开夹爪"""
        if not self.esp32_ip:
            print("未连接到设备，请先调用connect()")
            return False
        return self.send_command("OPEN")

    def close(self):
        """关闭夹爪"""
        if not self.esp32_ip:
            print("未连接到设备，请先调用connect()")
            return False
        return self.send_command("CLOSE")

    def disconnect(self):
        """断开与夹爪的连接"""
        if self.esp32_ip:
            print(f"断开与 {self.esp32_ip} 的连接")
            self.esp32_ip = None
        else:
            print("未连接到任何设备")
        return True

    def scan_network(self):
        """扫描局域网寻找ESP32设备，通过HTTP响应识别"""
        self.is_scanning = True
        print("正在扫描局域网中的ESP32设备...")
        self.esp32_ip = None

        # 获取本地IP和子网
        local_ip = self._get_local_ip()
        if not local_ip:
            print("无法获取本地IP地址，扫描失败")
            self.is_scanning = False
            return None

        # 解析子网
        subnet_parts = local_ip.split(".")[:3]
        subnet = ".".join(subnet_parts) + "."

        # 多线程扫描
        threads = []
        for i in range(1, 255):
            if self.esp32_ip:  # 已找到设备，停止扫描
                break

            ip = f"{subnet}{i}"
            thread = threading.Thread(target=self._check_ip_http, args=(ip,))
            threads.append(thread)
            thread.start()

            # 控制并发线程数
            if len(threads) >= MAX_THREADS:
                for t in threads:
                    t.join()
                threads = []

        # 等待剩余线程完成
        for t in threads:
            t.join()

        self.is_scanning = False

        if self.esp32_ip:
            print(f"扫描完成，找到ESP32设备: {self.esp32_ip}")
            return self.esp32_ip
        else:
            print("扫描完成，未找到ESP32设备")
            return None

    def get_gripper_status(self):
        """获取夹爪当前状态（辅助方法）"""
        return self.send_command("STATUS")

    def _get_local_ip(self):
        """获取本地IP地址（内部方法）"""
        try:
            # 通过UDP连接获取本地IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(('8.8.8.8', 80))
                return s.getsockname()[0]
        except Exception as e:
            print(f"获取本地IP出错: {e}")
            return None

    def _check_ip_http(self, ip):
        """检查指定IP是否为目标ESP32设备（内部方法）"""
        if self.esp32_ip:  # 已找到设备，直接返回
            return

        try:
            url = f"http://{ip}:{HTTP_PORT}/STATUS"
            response = requests.get(url, timeout=TIMEOUT)

            if response.status_code == 200:
                content = response.text.strip()
                if content.startswith("OK: "):
                    # 尝试解析角度值验证设备
                    try:
                        int(content[4:])
                        print(f"发现ESP32夹爪控制器: {ip}")
                        self.esp32_ip = ip
                    except ValueError:
                        pass
        except Exception:
            pass

    def send_command(self, command):
        """发送命令到ESP32设备（内部方法）"""
        if not self.esp32_ip:
            print("未连接到设备")
            return False

        # 限制命令发送频率
        current_time = time.time()
        if current_time - self.last_command_time < 0.5:
            print("命令发送过于频繁，请稍后再试")
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
                print(f"命令发送失败，HTTP状态码: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"无法连接到设备 {self.esp32_ip}，可能已离线")
            self.esp32_ip = None  # 标记为未连接，需要重新连接
            return False
        except Exception as e:
            print(f"发送命令时发生错误: {e}")
            return False

    def _verify_connection(self):
        """验证与指定IP的连接是否有效（内部方法）"""
        try:
            url = f"http://{self.esp32_ip}:{HTTP_PORT}/STATUS"
            response = requests.get(url, timeout=TIMEOUT)
            return response.status_code == 200 and response.text.strip().startswith("OK: ")
        except Exception:
            return False


# 主程序示例
if __name__ == "__main__":
    controller = GripperWiFiController()

    print("====== ESP32夹爪WiFi控制器 ======")

    # 演示核心方法的使用流程
    if not controller.connect():
        sys.exit(0)

    while True:
        print("\n====== 操作菜单 ======")
        print("1. 打开夹爪")
        print("2. 关闭夹爪")
        print("3. 查看夹爪状态")
        print("4. 重新连接设备")
        print("5. 断开连接并退出")

        choice = input("请选择操作 (1-5): ")

        if choice == "1":
            controller.open()
        elif choice == "2":
            controller.close()
        elif choice == "3":
            controller.get_gripper_status()
        elif choice == "4":
            controller.disconnect()
            controller.connect()
        elif choice == "5":
            controller.disconnect()
            print("程序退出")
            break
        else:
            print("无效的选择，请重新输入")
