import random
import socket
import threading
import time

import numpy as np


class DensorRobot:
    """
    Class for Robot sending and receiving
    命令指  含义                 示例
    SP#    发送位置数据           SP#300.0,0.0,200.0,-100.0,10.0,-90.0,1
    CP#    获取当前位置数据        CP#
    """

    static_send_position_prefix = "SP#"  # 发送位置数据指令类型
    static_current_position_prefix = "CP#"  # 获取当前位置数据指令类型
    static_relative_angle_prefix = "RA#"  # 发送相对角度旋转指令类型
    static_absolute_angle_prefix = "AA#"  # 发送绝对角度旋转指令类型

    def __init__(self, host="192.168.1.11", port=5002):
        self.host = host
        self.port = port
        self.receive_flag = True  # 用于控制接收消息的标识
        self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recv_thread = None
        self.current_position = []  # 机器臂当前位置

    def is_robot_connected(self):
        try:
            # 通过检查socket文件描述符判断连接状态
            return self.tcp_client is not None and self.tcp_client.fileno() != -1
        except:
            return False

    def connect(self) -> bool:
        try:
            # 设置超时时间
            self.tcp_client.settimeout(10)
            self.tcp_client.connect((self.host, self.port))
            self.recv_thread = threading.Thread(target=self.receive_message)  # 创建接收消息的线程
            self.recv_thread.daemon = True  # 设置线程为守护线程
            self.recv_thread.start()  # 启动接收消息线程
        except Exception as e:
            print(f"[robot] Connect failed: {e}")
            return False
        return True

    def close(self):
        self.receive_flag = False  # 设置接收消息的标识为False，结束接收线程
        if self.recv_thread:
            self.recv_thread.join(timeout=5)  # 等待接收线程结束
        if self.tcp_client:
            self.tcp_client.close()

    def __send_cmd(self, cmd):
        if not isinstance(cmd, str):
            raise TypeError(f"Expected 'cmd' to be of type str, but got {type(cmd)} instead.")
        # send command
        if len(cmd) > 0:
            if not cmd.endswith('\r'):  # 如果不是以\r结尾
                cmd += '\r'  # 拼接\r
            print('[robot] Sending message: ' + cmd)
            self.tcp_client.sendall(cmd.encode('utf-8'))

    def send_position(self, command):
        command_str = ''
        if isinstance(command, (list, np.ndarray)):  # 如果command是列表或NumPy数组
            command_str = ','.join(str(c) for c in command)  # 使用列表推导式将所有元素转换为字符串，并按","拼接
        # send command
        if len(command_str) > 0:
            # 先发一次指令类型
            self.__send_cmd(self.static_send_position_prefix)
            # 再发一次位置数据
            self.__send_cmd(command_str)

    def rotate_relative_angle(self, angle, j_num=6):
        """
        发送相对角度旋转 第六轴上，只针对平面抓取
        :param j_num: 旋转轴
        :param angle: 旋转角度, 单位为度
        """
        if angle is None:
            return
        command_str = f"{j_num},{angle}"
        # 先发一次指令类型
        self.__send_cmd(self.static_relative_angle_prefix)
        # 再发一次旋转数据
        self.__send_cmd(command_str)

    def rotate_absolute_angle(self, angle, j_num=6):
        """
        发送绝对角度旋转 第六轴上，只针对平面抓取
        :param j_num: 旋转轴
        :param angle: 旋转角度, 单位为度
        """
        if angle is None:
            return
        command_str = f"{j_num},{angle}"
        # 先发一次指令类型
        self.__send_cmd(self.static_absolute_angle_prefix)
        # 再发一次旋转数据
        self.__send_cmd(command_str)

    def get_current_position(self):
        """
        获取机器人当前位置
        :return:
        """
        self.__send_cmd(self.static_current_position_prefix)
        # 等待机器人返回当前位置信息
        time.sleep(1)
        return self.current_position

    def receive_message(self):
        while self.receive_flag:
            try:
                data = self.tcp_client.recv(1024)  # 接收消息
                msg = data.decode('utf-8')
                print("[robot] Receive message:", msg)  # 打印接收到的消息
                if msg.startswith(self.static_current_position_prefix):
                    # 去掉CP#以及开头可能存在的空格
                    msg = msg.replace("CP#", "").lstrip()
                    # 将字符串分割成列表，然后转换为float32
                    self.current_position = [float(num) for num in msg.split()]
            except Exception as e:  # 当socket连接出错时结束循环
                print("[robot] Socket error:", str(e))
                break


if __name__ == "__main__":
    # 机器人作为server端
    robot_ip = "192.168.1.11"
    robot_port = 5002
    robot = DensorRobot(host=robot_ip, port=robot_port)  # 传入机器人的IP地址和端口号

    # TOOL1 抓取初始化位置
    grap_init_position = [239.8948, 7.6883, 331.1265, -162.5392, 3.0513, 80.475, 5.0]
    # TOOL1 标定板初始化位置
    cal_init_position = [300.8917, -18.7299, 270.838, -144.9799, -69.0624, -36.25, 9.0]

    # home_position = [130.0, 0.0, 420.0, -167.69, 0.12, 82.23, 5]
    # time.sleep(2)
    # 打印当前位置
    print(robot.get_current_position())
    # # 回到默认位置
    # robot.send_position(home_position)
    # time.sleep(3)
    #
    # robot.rotate_relative_angle(-30, 6)
    # time.sleep(3)
    #
    # robot.rotate_relative_angle(30, 6)
    # time.sleep(3)
    #
    # robot.rotate_relative_angle(-30, 6)
    # time.sleep(3)
    #
    # robot.rotate_relative_angle(30, 6)
    # time.sleep(3)

    # i = 0
    # while i < 10:
    #     position = [home_position[0] + random.randint(5, 15) * (-1) ** i,
    #                 home_position[1] + random.randint(5, 15) * (-1) ** i,
    #                 home_position[2] + random.randint(5, 15) * (-1) ** i,
    #                 home_position[3],
    #                 home_position[4],
    #                 home_position[5],
    #                 home_position[6]]
    #     robot.send_position(position)  # 发送指令
    #     time.sleep(3)
    #     current_pos = robot.get_current_position()
    #     print("当前位置：", current_pos)
    #     i = i + 1
    robot.close()  # 关闭连接
