# coding=utf-8
'''
@Description: In User Settings Edit
@Author: Lai
@Date: 2019-11-09 12:23:59
LastEditTime: 2021-06-24 14:04:09
LastEditors: Qianen
'''
import time
import argparse
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
import numpy as np
from math import ceil

ROBOTIQ_IP = "192.168.1.111"


class Robotiq85(object):

    def __init__(self, MODBUS_SERVER_IP="192.168.1.103", timeout=2):
        print(MODBUS_SERVER_IP)
        self.client = None
        for _ in range(20):
            if not self.connectToDevice(MODBUS_SERVER_IP):
                print('正在重新连接到robotiq85...')
                time.sleep(0.2)
            else:
                break
        if not self.client.connect():
            raise Exception('gripper connect error')
        if self.is_reset():
            self.reset()
            self.activate(timeout)

    def connectToDevice(self, MODBUS_SERVER_IP):
        """ 连接到modbus """
        self.client = ModbusClient('192.168.1.103', 502)
        if not self.client.connect():
            print("Unable to connect to %s" % device)
            return False
        return True

    def disconnectFromDevice(self):
        """Close connection"""
        self.client.close()

    def _send_command(self, data):
        """ Send a command to the Gripper - the method takes a list of uint8 as an argument.
        The meaning of each variable depends on the Gripper model
        (see support.robotiq.com for more details)
        """
        # make sure data has an even number of elements
        if(len(data) % 2 == 1):
            data.append(0)

        # Initiate message as an empty list
        message = []

        # Fill message by combining two bytes in one register
        for i in range(int(len(data)/2)):
            message.append((data[2*i] << 8) + data[2*i+1])

        # To do!: Implement try/except
        self.client.write_registers(0x03E8, message, unit=0x0009)

    def _get_status(self, numBytes=6):
        """Sends a request to read, wait for the response and returns the Gripper status.
        The method gets the number of bytes to read as an argument"""
        numRegs = int(ceil(numBytes/2.0))

        # To do!: Implement try/except
        # Get status from the device
        response = self.client.read_holding_registers(0x07D0, numRegs, unit=0x0009)

        # Instantiate output as an empty list
        output = []

        # Fill the output with the bytes in the appropriate order
        for i in range(0, numRegs):
            output.append((response.getRegister(i) & 0xFF00) >> 8)
            output.append(response.getRegister(i) & 0x00FF)

        # Output the result
        return output

    def getStatus(self):
        """Request the status from the gripper and return it in the Robotiq2FGripper_robot_input msg type."""
        # Acquire status from the Gripper
        status = self._get_status(6)
        # Message to output
        message = {}
        # 夹爪是否正常工作
        message['gACT'] = (status[0] >> 0) & 0x01
        # 夹爪是否正在移动,移动时不能接收下一个指令
        message['gGTO'] = (status[0] >> 3) & 0x01
        message['gSTA'] = (status[0] >> 4) & 0x03
        # 是否检查到物体，0x00正在运动中没有物体，0x01物体在外面，0x02物体在里面，0x03运动到指定位置没有物体
        message['gOBJ'] = (status[0] >> 6) & 0x03
        # 错误信息
        message['gFLT'] = status[2]
        # 需要达到的位置，0x00全开，0xFF全闭
        message['gPR'] = status[3]
        # 当前位置
        message['gPO'] = status[4]
        # 10×电流=gCu(mA)
        message['gCU'] = status[5]
        return message

    def sendCommand(self, command):
        """ 把字典转化为可以直接发送的数组，并发送 """
        # 限制数值上下限
        for n in 'rACT rGTO rATR'.split():
            command[n] = int(np.clip(command.get(n, 0), 0, 1))
        for n in 'rPR rSP rFR'.split():
            command[n] = int(np.clip(command.get(n, 0), 0, 255))
        # 转换为要发送的数组
        message = []
        message.append(command['rACT'] + (command['rGTO'] << 3) + (command['rATR'] << 4))
        message.append(0)
        message.append(0)
        for n in 'rPR rSP rFR'.split():
            message.append(command[n])
        return self._send_command(message)

    def is_ready(self, status=None):
        status = status or self.getStatus()
        return status['gSTA'] == 3 and status['gACT'] == 1

    def is_reset(self, status=None):
        status = status or self.getStatus()
        return status['gSTA'] == 0 or status['gACT'] == 0

    def is_moving(self, status=None):
        status = status or self.getStatus()
        return status['gGTO'] == 1 and status['gOBJ'] == 0

    def is_stopped(self, status=None):
        status = status or self.getStatus()
        return status['gOBJ'] != 0

    def object_detected(self, status=None):
        status = status or self.getStatus()
        return status['gOBJ'] == 1 or status['gOBJ'] == 2

    def get_fault_status(self, status=None):
        status = status or self.getStatus()
        return status['gFLT']

    def get_pos(self, status=None):
        status = status or self.getStatus()
        po = status['gPO']
        # TODO:这里的变换还要调一下
        return np.clip(0.14/(13.-230.)*(po-230.), 0, 0.14)

    def get_req_pos(self, status=None):
        status = status or self.getStatus()
        pr = status['gPR']
        # TODO:这里的变换还要调一下
        return np.clip(0.14/(13.-230.)*(pr-230.), 0, 0.14)

    def is_closed(self, status=None):
        status = status or self.getStatus()
        return status['gPO'] >= 230

    def is_opened(self, status=None):
        status = status or self.getStatus()
        return status['gPO'] <= 13

    def get_current(self, status=None):
        status = status or self.getStatus()
        return status['gCU'] * 0.1

    def wait_until_stopped(self, timeout=None):
        start_time = time.time()
        while not self.is_reset() and (not timeout or (time.time() - start_time) < timeout):
            if self.is_stopped():
                return True
            time.sleep(0.1)
        return False

    def wait_until_moving(self, timeout=None):
        start_time = time.time()
        while not self.is_reset() and (not timeout or (time.time() - start_time) < timeout):
            if not self.is_stopped():
                return True
            time.sleep(0.1)
        return False

    def reset(self):
        cmd = {n: 0 for n in 'rACT rGTO rATR rPR rSP rFR'.split()}
        self.sendCommand(cmd)

    def activate(self, timeout=None):
        cmd = dict(rACT=1, rGTO=1, rATR=0, rPR=0, rSP=255, rFR=150)
        self.sendCommand(cmd)
        start_time = time.time()
        while not timeout or (time.time() - start_time) < timeout:
            if self.is_ready():
                return True
            time.sleep(0.1)
        return False

    def auto_release(self):
        cmd = {n: 0 for n in 'rACT rGTO rATR rPR rSP rFR'.split()}
        cmd['rACT'] = 1
        cmd['rATR'] = 1
        self.sendCommand(cmd)

    def goto(self, pos, vel=0.1, force=50, block=False, timeout=None):
        cmd = {n: 0 for n in 'rACT rGTO rATR rPR rSP rFR'.split()}
        cmd['rACT'] = 1
        cmd['rGTO'] = 1
        # cmd['rPR'] = int(np.clip((13.-230.)/0.14 * pos + 230., 0, 255))
        cmd['rPR'] = int(np.clip((0.-230.)/0.085 * pos + 230., 0, 255))
        cmd['rSP'] = int(np.clip(255./(0.1-0.013) * (vel-0.013), 0, 255))
        cmd['rFR'] = int(np.clip(255./(100.-30.) * (force-30.), 0, 255))
        self.sendCommand(cmd)
        time.sleep(0.1)
        if block:
            if not self.wait_until_moving(timeout):
                return False
            return self.wait_until_stopped(timeout)
        return True

    def stop(self, block=False, timeout=-1):
        cmd = {n: 0 for n in 'rACT rGTO rATR rPR rSP rFR'.split()}
        cmd['rACT'] = 1
        cmd['rGTO'] = 0
        self.sendCommand(cmd)
        time.sleep(0.1)
        if block:
            return self.wait_until_stopped(timeout)
        return True

    def open(self, vel=0.3, force=100, block=False, timeout=-1):
        if self.is_opened():
            return True
        return self.goto(1.0, vel, force, block=block, timeout=timeout)

    def close(self, vel=0.3, force=100, block=False, timeout=-1):
        if self.is_closed():
            return True
        return self.goto(-1.0, vel, force, block=block, timeout=timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='control robotiq85')
    parser.add_argument('-o', '--open', action='store_true',
                        help='open gripper')
    parser.add_argument('-c', '--close', action='store_true',
                        help='close gripper')
    parser.add_argument('-t', '--test', action='store_true',
                        help='test connection')
    parser.add_argument('-r', '--reset', action='store_true',
                        help='reset gripper')
    parser.add_argument('--ip', metavar=ROBOTIQ_IP, type=str, default=ROBOTIQ_IP,
                        help='server ip')
    args = parser.parse_args()
    gripper = Robotiq85(args.ip)
    gripper.activate()
    print("初始化完毕")
    #time.sleep(2)
    if args.reset:
        gripper.reset()
    if args.test:
        gripper.open()
        time.sleep(2)
        gripper.close()
        time.sleep(2)
    if args.open:
        gripper.open()
    elif args.close:
        gripper.close()
