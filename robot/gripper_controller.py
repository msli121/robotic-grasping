# -*- coding: utf-8 -*-
# @Time       : 2025/4/10 22:14
# @File       : gripper_controller.py
# @Description:
import asyncio
import logging
import os
import time
from bleak import BleakClient, BleakScanner
from typing import Optional

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BluetoothGripperController:
    """蓝牙夹爪控制器类，封装了夹爪的控制逻辑"""

    # 定义控制命令常量
    OPEN_GRIPPER_COMMAND = "#000P1900T1000!"
    CLOSE_GRIPPER_COMMAND = "#000P0950T1000!"
    SOFTWARE_RESET_COMMAND = "$RST!"

    def __init__(self, mac_address: str):
        self.mac_address = mac_address
        self._client: Optional[BleakClient] = None
        self._write_char_uuid: Optional[str] = None
        self._notify_char_uuid: Optional[str] = None
        self._is_connected = False
        self._loop = asyncio.new_event_loop()

    async def _ensure_connection(self) -> bool:
        if self._is_connected and self._client and self._client.is_connected:
            return True

        try:
            device = await BleakScanner.find_device_by_address(
                self.mac_address,
                cb=dict(use_bdaddr=False))
            if device is None:
                logger.error(f"无法找到地址为 {self.mac_address} 的设备")
                return False

            self._client = BleakClient(device)
            await self._client.connect()
            await self._discover_characteristics()

            self._is_connected = True
            logger.info(f"成功连接到夹爪设备 {self.mac_address}")
            return True

        except Exception as e:
            logger.error(f"连接设备时出错: {e}")
            self._is_connected = False
            return False

    async def connect(self) -> None:
        """显式连接设备，失败时抛出异常"""
        success = await self._ensure_connection()
        if not success:
            raise ConnectionError(f"无法连接到设备 {self.mac_address}")

    async def _discover_characteristics(self):
        if not self._client or not self._client.is_connected:
            return

        try:
            for service in self._client.services:
                for char in service.characteristics:
                    prop = char.properties
                    if "write" in prop:
                        self._write_char_uuid = char.uuid
                    if "notify" in prop:
                        self._notify_char_uuid = char.uuid

            if not self._write_char_uuid:
                logger.warning("未找到可写的特征UUID")
        except Exception as e:
            logger.error(f"发现特征时出错: {e}")

    async def _send_command(self, command: str) -> bool:
        if not await self._ensure_connection():
            return False

        if not self._write_char_uuid:
            logger.error("没有可用的写入特征UUID")
            return False

        try:
            await self._client.write_gatt_char(
                self._write_char_uuid,
                bytearray(command, "utf-8"),
                response=True
            )
            return True
        except Exception as e:
            logger.error(f"发送命令时出错: {e}")
            return False

    async def _execute_command(self, command: str, action_name: str) -> bool:
        success = await self._send_command(command)
        if success:
            await asyncio.sleep(1.5)
            logger.info(f"{action_name} 完成")
        return success

    async def open_gripper(self) -> bool:
        return await self._execute_command(
            self.OPEN_GRIPPER_COMMAND,
            "夹爪已张开"
        )

    async def close_gripper(self) -> bool:
        return await self._execute_command(
            self.CLOSE_GRIPPER_COMMAND,
            "夹爪已闭合"
        )

    async def software_reset(self) -> bool:
        return await self._execute_command(
            self.SOFTWARE_RESET_COMMAND,
            "软件复位"
        )

    async def set_gripper_position(self, position: int) -> bool:
        if position <= 100:
            logger.warning("位置值无效，请输入大于100的值")
            return False

        command = f"#000P0{position}T1000!"
        return await self._execute_command(
            command,
            f"夹爪已设置到位置 {position}"
        )

    async def disconnect(self) -> None:
        if not self._client or not self._client.is_connected:
            self._is_connected = False
            return

        try:
            # 停止通知前检查连接状态
            if self._notify_char_uuid and hasattr(self._client, 'stop_notify'):
                try:
                    if self._client.is_connected:
                        await self._client.stop_notify(self._notify_char_uuid)
                except Exception as e:
                    # 忽略"Not connected"错误，这表示连接已经断开
                    if "Not connected" not in str(e):
                        logger.error(f"停止通知时出错: {str(e)}")

            # 断开连接
            if self._client.is_connected:
                await self._client.disconnect()
                logger.info(f"已断开与设备 {self.mac_address} 的连接")

        except Exception as e:
            logger.error(f"断开连接时出错: {e}")
        finally:
            self._is_connected = False

    def __del__(self):
        try:
            if self._loop.is_running():
                # 创建一个新任务来安全地断开连接
                asyncio.run_coroutine_threadsafe(self.disconnect(), self._loop)
            else:
                self._loop.run_until_complete(self.disconnect())
        except Exception as e:
            logger.error(f"在析构函数中出错: {e}")
        finally:
            try:
                # 确保事件循环被正确关闭
                if not self._loop.is_closed():
                    self._loop.close()
            except Exception as e:
                logger.error(f"关闭事件循环时出错: {e}")


class GripperControllerWrapper:
    def __init__(self, mac_address: str = "EC:23:06:00:D9:FB"):
        self._controller = BluetoothGripperController(mac_address)
        self._loop = asyncio.new_event_loop()

    def connect(self) -> bool:
        """同步连接设备，失败时抛出异常"""
        try:
            self._loop.run_until_complete(self._controller.connect())
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False

    def open(self) -> bool:
        return self._run_async(self._controller.open_gripper())

    def close(self) -> bool:
        return self._run_async(self._controller.close_gripper())

    def reset(self) -> bool:
        return self._run_async(self._controller.software_reset())

    def set_gripper_position(self, position: int) -> bool:
        return self._run_async(self._controller.set_gripper_position(position))

    def disconnect(self) -> None:
        self._run_async(self._controller.disconnect())

    def _run_async(self, coro):
        try:
            # 如果事件循环未运行，则运行它
            if not self._loop.is_running():
                return self._loop.run_until_complete(coro)
            # 如果事件循环正在运行，则创建一个任务
            else:
                future = asyncio.run_coroutine_threadsafe(coro, self._loop)
                return future.result(timeout=10)  # 设置10秒超时
        except asyncio.TimeoutError:
            logger.error("操作超时")
            return False
        except Exception as e:
            logger.error(f"执行操作时出错: {e}")
            return False

    def __del__(self):
        self.disconnect()
        self._loop.close()


if __name__ == "__main__":
    MAC_ADDRESS = "EC:23:06:00:D9:FB"

    # # 创建事件循环策略以避免Windows上的问题
    # if os.name == 'nt':
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    gripper = GripperControllerWrapper(MAC_ADDRESS)

    try:
        # 尝试连接，最多尝试3次
        connected = False
        for i in range(3):
            try:
                gripper.connect()
                connected = True
                break
            except ConnectionError as e:
                logger.warning(f"连接尝试 {i+1} 失败: {e}")
                if i < 2:  # 如果不是最后一次尝试，等待2秒再试
                    time.sleep(2)
        
        if not connected:
            logger.error("无法连接到夹爪设备")
            exit(1)
            
        gripper.open()
        time.sleep(2)
        gripper.close()
        time.sleep(2)
        # gripper.reset()
        # time.sleep(2)
        print("gripper closed")
    except ConnectionError as e:
        logger.error(f"连接异常: {e}")
    except Exception as e:
        logger.error(f"发生错误: {e}")
    finally:
        try:
            gripper.disconnect()
        except Exception as e:
            logger.error(f"断开连接时出错: {e}")
