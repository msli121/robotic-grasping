# -*- coding: utf-8 -*-
# @Time       : 2025/4/10 22:14 (修复版)
# @File       : gripper_controller_fixed.py
# @Description: 修复线程竞态问题并优化日志管理的蓝牙夹爪控制器

import asyncio
import logging
import threading
import time
from bleak import BleakClient, BleakScanner
from typing import Optional

# --- 1. 优化日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("bleak").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class BluetoothGripperController:
    """(异步)蓝牙夹爪控制器 - 内部类"""

    OPEN_GRIPPER_COMMAND = b"#000P1900T1000!"
    CLOSE_GRIPPER_COMMAND = b"#000P0950T1000!"
    SOFTWARE_RESET_COMMAND = b"$RST!"

    def __init__(self, mac_address: str):
        self.mac_address = mac_address
        self._client: Optional[BleakClient] = None
        self._write_char_uuid: Optional[str] = None
        self._is_connected = False

    async def connect(self) -> bool:
        if self._is_connected and self._client and self._client.is_connected:
            return True

        logger.info(f"正在扫描设备: {self.mac_address}...")
        try:
            device = await BleakScanner.find_device_by_address(self.mac_address, timeout=10.0)
            if not device:
                logger.error(f"无法找到设备 {self.mac_address}")
                return False

            logger.info("找到设备，正在连接...")
            self._client = BleakClient(device)
            await self._client.connect()

            await self._discover_characteristics()

            self._is_connected = self._client.is_connected
            if self._is_connected:
                logger.info(f"成功连接到夹爪设备 {self.mac_address}")
            return self._is_connected
        except Exception as e:
            logger.error(f"连接设备时出错: {e}", exc_info=True)
            self._is_connected = False
            return False

    async def _discover_characteristics(self):
        if not self._client:
            return
        for service in self._client.services:
            for char in service.characteristics:
                if "write" in char.properties:
                    self._write_char_uuid = char.uuid
                    logger.debug(f"发现可写特征: {self._write_char_uuid}")
                    return
        logger.warning("警告：未找到任何可写的特征UUID！")

    async def disconnect(self) -> None:
        if self._client and self._client.is_connected:
            try:
                await self._client.disconnect()
                logger.info("已断开连接。")
            except Exception as e:
                logger.error(f"断开连接时出错: {e}", exc_info=True)
        self._is_connected = False

    async def _send_command(self, command: bytes, action_name: str) -> bool:
        if not (self._is_connected and self._client and self._write_char_uuid):
            logger.error(f"无法执行'{action_name}'：设备未连接或写入特征不可用。")
            return False
        try:
            await self._client.write_gatt_char(self._write_char_uuid, command, response=True)
            logger.debug(f"命令 '{action_name}' 已发送。")
            await asyncio.sleep(1.5)
            logger.debug(f"'{action_name}' 操作完成（固定延迟结束）。")
            return True
        except Exception as e:
            logger.error(f"发送命令 '{action_name}' 时出错: {e}", exc_info=True)
            self._is_connected = False
            return False

    async def open_gripper(self) -> bool:
        return await self._send_command(self.OPEN_GRIPPER_COMMAND, "夹爪张开")

    async def close_gripper(self) -> bool:
        return await self._send_command(self.CLOSE_GRIPPER_COMMAND, "夹爪闭合")

    async def software_reset(self) -> bool:
        return await self._send_command(self.SOFTWARE_RESET_COMMAND, "软件复位")


class GripperControllerWrapper:
    """同步接口包装器（修复版）"""

    def __init__(self, mac_address: str = "EC:23:06:00:D9:FB"):
        self._controller = BluetoothGripperController(mac_address)
        self._loop = asyncio.new_event_loop()

        # <<< FIX 1: 创建一个 threading.Event 用于线程同步
        self._loop_started = threading.Event()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # <<< FIX 2: 等待，直到后台线程发信号说事件循环已准备就绪
        self._loop_started.wait()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        # <<< FIX 3: 在启动循环前，发信号通知主线程可以继续了
        self._loop_started.set()
        try:
            self._loop.run_forever()
        finally:
            # 清理循环中的挂起任务
            tasks = asyncio.all_tasks(loop=self._loop)
            for task in tasks:
                task.cancel()
            group = asyncio.gather(*tasks, return_exceptions=True)
            self._loop.run_until_complete(group)
            self._loop.close()

    def _run_sync(self, coro) -> any:
        if not self._loop.is_running():
            logger.error("事件循环未运行，无法执行命令。")
            return False

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=15.0)
        except asyncio.TimeoutError:
            logger.error("操作超时！任务在15秒内未能完成。")
            return False
        except Exception as e:
            logger.error(f"执行异步任务时出错: {e}", exc_info=True)
            return False

    def connect(self) -> bool:
        return self._run_sync(self._controller.connect())

    def open(self) -> bool:
        return self._run_sync(self._controller.open_gripper())

    def close(self) -> bool:
        return self._run_sync(self._controller.close_gripper())

    def reset(self) -> bool:
        return self._run_sync(self._controller.software_reset())

    def disconnect(self) -> None:
        self._run_sync(self._controller.disconnect())

    def shutdown(self):
        if self._loop.is_running():
            logger.info("正在关闭控制器...")
            self.disconnect()
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("后台线程未能正常关闭。")
            logger.info("控制器已关闭。")

    def __del__(self):
        if hasattr(self, '_loop') and self._loop.is_running():
            self.shutdown()


if __name__ == "__main__":
    MAC_ADDRESS = "EC:23:06:00:D9:FB"

    # 切换日志级别以查看详细调试信息
    # logger.setLevel(logging.DEBUG)

    gripper = GripperControllerWrapper(MAC_ADDRESS)

    try:
        # 主线程现在会等到事件循环就绪后才调用 connect
        logger.info("开始连接夹爪...")
        connected = False
        for i in range(3):
            if gripper.connect():
                connected = True
                break
            else:
                logger.warning(f"连接尝试 {i + 1}/3 失败，2秒后重试...")
                time.sleep(2)

        if not connected:
            logger.error("三次尝试后仍无法连接到夹爪设备，程序退出。")
            exit(1)

        logger.info("连接成功！开始执行开合测试...")
        for i in range(2):
            logger.info(f"--- 第 {i + 1} 次测试 ---")
            gripper.open()
            gripper.close()
        logger.info("测试完成。")

    except Exception as e:
        logger.error(f"主程序发生未知错误: {e}", exc_info=True)
    finally:
        logger.info("正在关闭程序...")
        gripper.shutdown()
        logger.info("程序已退出。")