# -*- coding: utf-8 -*-
# @Time       : 2025/4/10 22:14
# @File       : gripper_controller.py
# @Description:
import asyncio
import logging
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
            logger.info(f"成功连接到设备 {self.mac_address}")
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
        if not self._client:
            return

        for service in self._client.services:
            for char in service.characteristics:
                prop = char.properties
                if "write" in prop:
                    self._write_char_uuid = char.uuid
                if "notify" in prop:
                    self._notify_char_uuid = char.uuid

        if not self._write_char_uuid:
            logger.warning("未找到可写的特征UUID")

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
        if not self._client:
            return

        try:
            if self._notify_char_uuid and hasattr(self._client, 'stop_notify'):
                try:
                    await self._client.stop_notify(self._notify_char_uuid)
                except Exception as e:
                    logger.error(f"停止通知时出错: {str(e)}")

            if self._client.is_connected:
                await self._client.disconnect()
                logger.info(f"已断开与设备 {self.mac_address} 的连接")

        except Exception as e:
            logger.error(f"断开连接时出错: {e}")
        finally:
            self._is_connected = False

    def __del__(self):
        if self._loop.is_running():
            self._loop.create_task(self.disconnect())
        else:
            self._loop.run_until_complete(self.disconnect())
        self._loop.close()


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
            return self._loop.run_until_complete(coro)
        except Exception as e:
            logger.error(f"执行操作时出错: {e}")
            return False

    def __del__(self):
        self.disconnect()
        self._loop.close()


if __name__ == "__main__":
    MAC_ADDRESS = "EC:23:06:00:D9:FB"

    gripper = GripperControllerWrapper(MAC_ADDRESS)

    try:
        gripper.connect()  # 显式连接，失败会抛出异常
        gripper.open()
        time.sleep(2)
        gripper.close()
        time.sleep(2)
        # gripper.reset()
        # time.sleep(2)
        print("gripper closed")
        gripper.disconnect()
    except ConnectionError as e:
        logger.error(f"连接异常: {e}")
    finally:
        gripper.disconnect()
