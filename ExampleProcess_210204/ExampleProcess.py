# _var are Variables which are only valid for a single loop iteration and they are local variables only

# general imports
import json
import numpy as np
import math
import threading
from matplotlib import pyplot as plt
from time import time, sleep
import tkinter as tk
from tkinter import messagebox
from scipy import interpolate
from datetime import datetime
import random

simMode = 1  # simulation mode to test basic function of the script (no Hardware is simulated)
log = 1  # 0..only basic state comments, 1..activate all logging

if not simMode:
    import oma
    import omax
    ZAxis = oma.AccAx('axisZ')


def main():

    ts_SM = 0.05  # sample time state machine thread
    ts_DC = 0.05  # sample time data collector thread

    # Define Process
    rr = 20  # ramp-rate in deg/s

    T1 = 148  # set temperature is know up front
    T2 = 178
    T3 = 148
    t0 = 2
    t1 = 3  # hold T1 for 60sec
    t2 = 4
    t3 = 5

    if simMode:
        T0 = 18  # start temperature should be read before creation of the vector

    # Z-Axis configuration
    spd = 10
    acc = 1000
    minPos = 0
    maxPos = 10.5
    CuringPos = 6.0
    safetyPos = 0.0
    PreCuringPos = 2.0

    # Heaters
    Kp_th = 150
    Ki_th = 0.8
    iLim_th = 100
    Kaw_th = Ki_th/Kp_th*50.0e-3
    Kp_bh = 100
    Ki_bh = 0.5
    iLim_bh = 100
    Kaw_bh = Ki_bh/Kp_bh*50.0e-3

    # Force, Air pressure
    fixPressureSetVal = 5233
    varPressureSetVal = 5870  # 10N ... 5820mbar

    # Init OMA objects
    if not simMode:
        oTh = Heater('HT_CT_02', 'HT_TEMP2_02')
        oBh = Heater('HT_CT_00', 'HT_TEMP2_00')
        oTh.setCalibration([1480, 1780], [1360, 1650])
        oBh.setCalibration([1480, 1780], [1395, 1660])
        oCool = oma.DigitalOut('Cooling')

        # analog out pressure
        oForce = Force(nameAO_fixPressure='const_pressure_set', nameAO_varPressure='var_pressure_set',
                       sensorPort_fixPressure='USB3.3;9600', sensorPort_varPressure='USB3.2;9600')

    # start data collector in its own thread
    if not simMode:
        dc = DataCollector(ts=ts_DC, oThermode=oTh, oBHeater=oBh, ZAxis=ZAxis, oForce=oForce, oCool=oCool)
    else:
        dc = DataCollector(ts=ts_DC, oThermode=[], oBHeater=[], ZAxis=[], oForce=[], oCool=[])

    dc.start()  # start second thread
    sleep(0.05)

    # init variables
    prevState = 'none'
    state = 'Init'
    flagEnterState = 1  # 1...state is being entered, 0...already in state
    vState = []
    _Tset = 0.0  # current set temperature
    vTsetState = []
    vDt = []
    tStartWhile = time()
    _tRel = time() - tStartWhile
    _tOld = 0.0

    # main loop which is executed every ts,  _var notation refers to values in the current loop iteration
    while _tRel < 1200.0:  # limit loop time to 10min
        _tAbs = time()  # get absolute system time
        _tRel = _tAbs - tStartWhile  # get relative time since loop start
        _dt = _tRel - _tOld
        vDt.append(_dt)
        _tOld = _tRel
        if not simMode:
            _Threal = oTh.getHeaterTemp()
            _Tbreal = oBh.getHeaterTemp()
        else:
            _Threal = 0.0
            _Tbreal = 0.0

        # STATE MACHINE
        if state == 'Init':

            if flagEnterState:
                nextState = 'PreCuringPos'  # TODO: Only for testing
                print('state ' + state + ' entered at time: ' + str(_tRel) + ' sec')
                vState.append((state, _tAbs))
                flagEnterState = 0  # set flag for state entered

                if not simMode:
                    # Init Z-Axis
                    ok = initZAxis(spd, acc, minPos, maxPos)
                    if ok == -1:  # if Z-Axis init failed, jump to END state
                        state = 'FAIL'
                        if log:
                            print('Z-Axis init failed')
                        continue
                    # Init Heaters
                    ok = oTh.setParameters(Kp_th, Ki_th, iLim_th, Kaw_th)
                    if ok == -1:
                        state = 'FAIL'
                        if log:
                            print(
                                'Failed to set these parameters: Kp:' + f'{Kp_th:.1f}' + 'Ki:' f'{Ki_th:.4f}' + 'iLim:' f'{iLim_th:.0f}' + 'Kaw:' f'{Kaw_th:.4f}')
                        continue
                    elif ok == 1:
                        if log:
                            print(
                                'Parameters successfully set: Kp:' + f'{Kp_th:.1f}' + 'Ki:' f'{Ki_th:.4f}' + 'iLim:' f'{iLim_th:.0f}' + 'Kaw:' f'{Kaw_th:.4f}')
                    ok = oBh.setParameters(Kp_bh, Ki_bh, iLim_bh, Kaw_bh)
                    if ok == -1:
                        state = 'FAIL'
                        if log:
                            print(
                                'Failed to set these parameters: Kp:' + f'{Kp_bh:.1f}' + 'Ki:' f'{Ki_bh:.4f}' + 'iLim:' f'{iLim_bh:.0f}' + 'Kaw:' f'{Kaw_bh:.4f}')
                        continue
                    elif ok == 1:
                        if log:
                            print(
                                'Parameters successfully set: Kp:' + f'{Kp_th:.1f}' + 'Ki:' f'{Ki_th:.4f}' + 'iLim:' f'{iLim_th:.0f}' + 'Kaw:' f'{Kaw_th:.4f}')
                    # Power on heaters
                    ok = oTh.On()
                    if ok == -1:
                        state = 'END'
                        if log:
                            print('Thermode heater power on failed')
                    ok = oBh.On()
                    if ok == -1:
                        state = 'END'
                        if log:
                            print('Bottom heater power on failed')
                    # Init Air pressure
                    oForce.Off()
                    sleep(0.5)
                    oForce.setPressure(fixPressureSetVal=fixPressureSetVal, varPressureSetVal=varPressureSetVal)
                    sleep(1)
                    _fixPressure, _varPressure = oForce.getPressure()
                    _fixPressure = _fixPressure._convert_magnitude('mbar')
                    _varPressure = _varPressure._convert_magnitude('mbar')
                    if abs(_fixPressure - fixPressureSetVal) > fixPressureSetVal * 0.2:
                        state = 'FAIL'
                        if log:
                            print('Constant pressure exceeds more than 20% of its set-value')
                    else:
                        if log:
                            print('Constant and variable pressure set successful')
                    if abs(_varPressure - varPressureSetVal) > varPressureSetVal * 0.2:
                        state = 'FAIL'
                        if log:
                            print('Variable pressure exceeds more than 20% of its set-value')
                    else:
                        if log:
                            print('Variable pressure set successful')
                else:
                    _Tset = T0
                    sleep(3)

            if 1:
                prevState = state  # save state history
                state = nextState  # overwrite given state to jump to the next state
                flagEnterState = 1  # reset flag for next state

        # Move Z-Axis to pre-curing position
        elif state == 'PreCuringPos':
            if flagEnterState:
                if prevState == 'Init':
                    nextState = 'HoldTemp'
                if prevState == 'HoldTemp':
                    nextState = 'Cool'
                timeout = 30
                print('state ' + state + ' entered at time: ' + str(_tRel) + ' sec')
                vState.append((state, _tAbs))
                flagEnterState = 0  # set flag for state entered
                tStartState = time()  # set time of state entry
                if log:
                    print('Move Z-Axis to pre-curing-position')
                if not simMode:
                    ZAxis.cmdGoPos(PreCuringPos)

            if not simMode:
                _M60 = ZAxis.ETLcmd('M60')
            else:
                _M60 = 0
                sleep(2)

            if not _M60 & 16 == 16:  # monitor moving bit
                prevState = state  # save state history
                state = nextState  # overwrite given state to jump to the next state
                flagEnterState = 1  # reset flag for next state
            elif time() - tStartState >= timeout:
                prevState = state  # save state history
                state = 'FAIL'
                print('Move Z-Axis to pre-curing-position failed')
                continue

        # Ramp ambient to 148 deg with 7deg/s
        elif state == 'Ramp1':

            if flagEnterState:
                Tend = T1
                if not simMode:
                    Tstart = _Threal
                else:
                    Tstart = T0
                endTime = (Tend - Tstart)/rr
                #fcnT01 = linearRamp(endTime, ts_SM, Tstart, Tend)  # create Temp ramp as function of time
                fcnT01 = sigmoidRamp(endTime, ts_SM, Tstart, Tend)  # create Temp ramp as function of time
                nextState = 'HoldTemp'
                print('state ' + state + ' entered at time: ' + str(_tRel) + ' sec')
                vState.append((state, _tAbs))
                flagEnterState = 0  # set flag for state entered
                tStartState = time()  # set time of state entry
                #ZAxis.cmdGoPos(HeaterPos)

            _tState = time() - tStartState
            if _tState <= endTime:  # only increase set temperature if theoretical end-time is not exceeded
                _Tset = float(fcnT01(_tState))
                if not simMode:
                    oTh.setSetTemp(_Tset)
                    oBh.setSetTemp(_Tset)
                if log:
                    print(state + ', Tset: ' + f'{_Tset:.1f}' + ', T: ' + f'{_Threal:.1f}')

            if _tState >= endTime:
                prevState = state  # save state history
                state = nextState  # overwrite given state to jump to the next state
                flagEnterState = 1  # reset flag for next state

        # State Hold Temperature
        elif state == 'HoldTemp':
            # basically do nothing till the timer elapses
            if flagEnterState:
                if prevState == 'PreCuringPos':  # Hold Room-temp after moving to pre-curing position
                    endTime = t0
                    nextState = 'Ramp1'
                    if not simMode:
                        _Tset = _Threal
                    else:
                        _Tset = T0
                if prevState == 'Ramp1':  # Hold 148deg after ramping up
                    endTime = t1

                    nextState = 'InsertLid'
                    #nextState = 'Cool'
                    _Tset = T1
                elif prevState == 'CuringPos':  # Hold 148deg after touchdown
                    endTime = t1
                    nextState = 'Ramp2'
                    _Tset = T1
                elif prevState == 'Ramp2':  # Hold 178deg after ramping up
                    endTime = t2
                    nextState = 'PreCuringPos'
                    _Tset = T2
                elif prevState == 'Cool':  # Hold 148deg after cooling down
                    endTime = t3
                    nextState = 'END'
                    _Tset = T3
                tStartState = time()  # set time of state entry
                if not simMode:
                    oTh.setSetTemp(_Tset)
                    oBh.setSetTemp(_Tset)
                print('state ' + state + ' entered at time: ' + str(_tRel) + ' sec')
                vState.append((state, _tAbs))
                flagEnterState = 0  # set flag for state entered

            if log:
                print(
                    'HoldTemp, ' + 'Tset: ' + f'{_Tset:.0f}' + 'deg, t-' + f"{endTime - (time() - tStartState):.2f}" + 's')

            if time() - tStartState >= endTime:
                prevState = state  # save state history
                state = nextState  # overwrite given state to jump to the next state
                flagEnterState = 1  # reset flag for next state

        # Insert Lid
        elif state == 'InsertLid':

            if flagEnterState:
                nextState = 'CuringPos'
                #ZAxis.cmdGoPos(PreCuringPos)
                print('state ' + state + ' entered at time: ' + str(_tRel) + ' sec')
                vState.append((state, _tAbs))
                flagEnterState = 0  # set flag for state entered

            answer = ShowMessageYesNo(
                'Please insert the Lid onto the bottom-heater. Press "Yes" to proceed or "No" to abort!', 'Insert Lid')

            if answer:  # answer:
                prevState = state  # save state history
                state = nextState  # overwrite given state to jump to the next state
                flagEnterState = 1  # reset flag for next state
                if log:
                    print(state + ', Message box confirmed')
            elif not answer:
                prevState = state  # save state history
                state = 'END'  # jump to END state
                flagEnterState = 1  # reset flag for next state
                if log:
                    print(state + ', Message box declined --> jump to END')

        # Move Z-Axis to curing position
        elif state == 'CuringPos':

            if flagEnterState:
                nextState = 'HoldTemp'
                print('state ' + state + ' entered at time: ' + str(_tRel) + ' sec')
                vState.append((state, _tAbs))
                flagEnterState = 0  # set flag for state entered
                timeout = 30
                tStartState = time()  # set time of state entry
                if log:
                    print('Move Z-Axis to curing-position')
                if not simMode:
                    ZAxis.cmdGoPos(CuringPos)

            if not simMode:
                _M60 = ZAxis.ETLcmd('M60')
            else:
                _M60 = 0
                sleep(2)

            if not _M60 & 16 == 16:  # monitor moving bit
                prevState = state  # save state history
                state = nextState  # overwrite given state to jump to the next state
                flagEnterState = 1  # reset flag for next state
            elif time() - tStartState >= timeout:
                prevState = state  # save state history
                state = 'FAIL'
                print('Move Z-Axis to curing-position failed')
                continue

        # Ramp from 148deg to 178deg with 7deg/s
        elif state == 'Ramp2':

            if flagEnterState:
                if not simMode:
                    Tstart = _Threal
                else:
                    Tstart = T1
                Tend = T2
                endTime = (Tend - Tstart)/rr
                #fcnT12 = linearRamp(endTime, ts_SM, Tstart, Tend)  # create Temp ramp as function of time
                fcnT12 = sigmoidRamp(endTime, ts_SM, Tstart, Tend)  # create Temp ramp as function of time
                nextState = 'HoldTemp'
                print('state ' + state + ' entered at time: ' + str(_tRel) + ' sec')
                vState.append((state, _tAbs))
                flagEnterState = 0  # set flag for state entered
                tStartState = time()  # set time of state entry

            _tState = time() - tStartState
            if _tState <= endTime:  # only increase set temperature if theoretical end-time is not exceeded
                _Tset = float(fcnT12(_tState))
                if not simMode:
                    oTh.setSetTemp(_Tset)
                    oBh.setSetTemp(_Tset)
                if log:
                    print(state + ', Tset: ' + f'{_Tset:.1f}' + ', T: ' + f'{_Threal:.1f}')

            if _tState >= endTime:
                prevState = state  # save state history
                state = nextState  # overwrite given state to jump to the next state
                flagEnterState = 1  # reset flag for next state

        # Cool down to 148deg
        elif state == 'Cool':
            if flagEnterState:
                nextState = 'HoldTemp'
                _Tset = T3
                if not simMode:
                    oTh.setSetTemp(_Tset)
                    oBh.setSetTemp(_Tset)
                    oCool.cmdSet()  # activate cooling
                print('state ' + state + ' entered at time: ' + str(_tRel) + ' sec')
                vState.append((state, _tAbs))
                flagEnterState = 0  # set flag for state entered

            if not simMode:
                trE = _Tset - _Threal
            else:
                trE = 0.0
                sleep(3)

            if log:
                print(state + ', Tset: ' + f'{_Tset:.1f}' + ', T: ' + f'{_Threal:.1f}')

            if abs(trE) <= 1:  # if tracking error is smaller than +/-1deg, cool down is finished
                if not simMode:
                    oCool.cmdReset()
                prevState = state  # save state history
                state = nextState  # overwrite given state to jump to the next state
                flagEnterState = 1  # reset flag for next state

        # End state machine
        elif state == 'END':
            print('state ' + state + ' entered at time: ' + str(_tRel) + ' sec')
            vState.append((state, _tAbs))

            if not simMode:
                ZAxis.cmdGoPos(safetyPos)
                ZAxis.cmdPwr(False)  # Power off z-Axis
                oTh.Off()
                oBh.Off()
                oForce.Off()
            else:
                sleep(2)
            vState.append(('', time()))  # append additional state for Measurement end
            dc.stop()  # stop data collector thread
            break

        # Failure state
        elif state == 'FAIL':
            print('state ' + state + ' entered at time: ' + str(_tRel) + ' sec')
            vState.append((state, _tAbs))

            if not simMode:
                ZAxis.cmdPwr(False)  # Power off z-Axis
                oTh.Off()
                oBh.Off()
                oForce.Off()
            else:
                sleep(2)
            vState.append(('', time()))  # append additional state for Measurement end
            dc.stop()  # stop data collector thread
            break

        # Catch state
        else:
            print('state not defined')
            state = 'END'

        if not simMode:
            # Check for ETEL Axis in error
            _M60 = ZAxis.ETLcmd('M60')
            if _M60 & 1024 == 1024:  # M60 bit 10: Controller in error status
                state = 'FAIL'
                continue
        else:
            vTsetState.append((_tAbs, _Tset))

        # pause if sample time has not elapsed yet
        sleep(ts_SM - time() % ts_SM)

        # ******************** MAIN LOOP END ******************** #

    # Remove start time from absolute time values to synchronize thread times
    vState = [x[1] - dc.data['Time'][0] for x in vState]
    dc.data['Time'] = [x - dc.data['Time'][0] for x in dc.data['Time']]
    # Append vectors to data
    dc.data['State'] = vState
    dc.data['dt_StateMachine'] = vDt
    dc.write_to_json()

    if simMode:
        plt.figure(0)
        plt.plot(*zip(*vTsetState))

    # Plot collected data
    '''dc.plotTemp()
    dc.plotActuatingVars()
    plt.show()'''


# ******************** MAIN() END ******************** #


# ******************** FUNCTIONS ******************** #

def initZAxis(spd, acc, minPos, maxPos):
    global log
    ZAxis.MinPos = minPos
    ZAxis.MaxPos = maxPos
    ZAxis.cmdRst()
    if log:
        print('restting Z-Axis')

    # Take Axis power on
    ZAxis.cmdPwr(True)
    _M60 = ZAxis.ETLcmd('M60')
    tt = time()  # set start time for timeout
    while not _M60 & 1 == 1:  # wait for axis power on (bit0)
        _M60 = ZAxis.ETLcmd('M60')
        if tt - time() > 10:  # timeout t>10sec
            print('Z-Axis power on failed')
            return -1
    if log:
        print('Z-Axis power on')

    # Home Axis
    ZAxis.cmdInd(9)
    tt = time()  # set start time for timeout
    while not _M60 & 4 == 4:  # wait for axis homed (bit2)
        _M60 = ZAxis.ETLcmd('M60')
        if tt - time() > 10:  # timeout t>10sec
            print('Z-Axis homing failed')
            return -1
    if log:
        print('Z-Axis successfully homed')
    ZAxis.Speed = spd
    ZAxis.Acc = acc
    if log:
        print('Z-Axis initiated')

    return 1


def ShowMessageYesNo(msg, title=None):
    print(msg)
    if title is None:
        title = msg

    root = tk.Tk()
    root.overrideredirect(1)
    root.withdraw()
    root.attributes("-topmost", True)
    result = messagebox.askyesno(title, msg)  # , icon='warning')#, parent=texto)
    root.destroy()
    return result


class Force:

    def __init__(self, nameAO_fixPressure, nameAO_varPressure, sensorPort_fixPressure, sensorPort_varPressure):
        self.AO_fix_pressure = oma.AnalogOut(Name=nameAO_fixPressure)
        self.AO_var_pressure = oma.AnalogOut(Name=nameAO_varPressure)
        self.get_fix_pressure = omax.OmaPressureSensor(Port=sensorPort_fixPressure)
        self.get_var_pressure = omax.OmaPressureSensor(Port=sensorPort_varPressure)

    def Off(self):
        self.AO_var_pressure.Value = 0
        if log:
            print('Variable Air pressure turned off...')
        self.AO_fix_pressure.Value = 0
        if log:
            print('Constant Air pressure turned off...')

    def setPressure(self, fixPressureSetVal, varPressureSetVal):
        self.AO_fix_pressure.Value = fixPressureSetVal
        if log:
            print('Constant Air pressure ' + f'{fixPressureSetVal:.4f}' + ' turned on...')
        self.AO_var_pressure.Value = varPressureSetVal
        if log:
            print('Variable Air pressure ' + f'{varPressureSetVal:.4f}' + ' turned off...')

    def getPressure(self):
        _fixPressure = self.get_fix_pressure.getPressure()
        _varPressure = self.get_var_pressure.getPressure()

        return _fixPressure, _varPressure


def linearRamp(tend, tsample, Tstart, Tend):
    # create Ramp vectors
    n01 = np.int(np.round(tend / tsample))
    vt01 = np.linspace(0, tend, n01)
    vT01 = np.linspace(Tstart, Tend, n01)

    # extend Tend flatline by a 10% to avoid out-of-bounds
    tExt = tend * 1.1
    nExt = np.int(np.round((tExt - tend) / tsample))
    vtExt = np.linspace(tend, tExt, nExt)
    vTExt = np.linspace(Tend, Tend, nExt)

    vt = np.append(vt01, vtExt)
    vT = np.append(vT01, vTExt)
    fcnT01 = interpolate.interp1d(vt, vT)

    return fcnT01


def tanhRamp(tend, tsample, Tstart, Tend):
    n01 = np.int(np.round(tend / tsample))
    x0 = 0.0
    xE = 2 * np.pi
    x = np.linspace(x0, xE, n01)

    y = []
    for i in range(0, len(x)):
        y.append(math.tanh(x[i]))
    x = [k / (xE - x0) * tend + tend / 2 for k in x]
    y = [Tstart + k * (Tend - Tstart) for k in y]
    # plt.plot(x,y)
    fcnT = interpolate.interp1d(x, y)

    return fcnT


def sigmoidRamp(tend, tsample, Tstart, Tend):
    n01 = np.int(np.round(tend / tsample))
    x0 = -2 * np.pi
    xE = 2 * np.pi
    x = np.linspace(x0, xE, n01)

    y = []
    for i in range(0, len(x)):
        y.append(1 / (1 + math.exp(-x[i])))
    x = [k / (xE - x0) * tend + tend / 2 for k in x]
    y = [Tstart + k * (Tend - Tstart) for k in y]
    # plt.plot(x,y)
    fcnT = interpolate.interp1d(x, y)

    return fcnT


class Heater:
    # Wrapper class for heater

    def __init__(self, name, nameAI):
        self.name = name
        self.nameAI = nameAI
        self.Kp = []
        self.Ki = []
        self.iLim = []
        self.Kaw = []
        self.factor = []
        self.offset = []
        self.oCtrl = oma.Ctrl(self.name)  # initiate oma object
        self.oAI = oma.AnalogIn(nameAI)

    def setParameters(self, Kp, Ki, iLim, Kaw):
        self.Kp = Kp
        self.Ki = Ki
        self.iLim = iLim
        self.Kaw = Kaw

        self.oCtrl.setPID(self.Kp, self.Ki, 0.0)
        self.oCtrl.setAntiWindupIntegrationLimit(iLim)
        self.oCtrl.setAntiWindupGain(Kaw)

        self.Kp = self.getParam('Kp')
        self.Ki = self.getParam('Ki')
        self.iLim = self.getParam('iLim')
        self.Kaw = self.getParam('Kaw')
        tol = 0.001  # tolerance for values: 0.1%
        if self.Kp - Kp <= Kp*tol and self.Ki-Ki <= Ki*tol and self.iLim - iLim <= iLim*tol and self.Kaw - Kaw <= Kaw*tol:
            if log:
                print(
                    'Setting of variables Kp:' + f'{self.Kp:.1f}' + 'Ki:' f'{self.Ki:.4f}' + 'iLim:' f'{self.iLim:.0f}' + 'Kaw:' f'{self.Kaw:.4f}')
            return 1
        else:
            print(
                'Failed to set parameters. Returned parameters are Kp:' + f'{self.Kp:.1f}' + 'Ki:' f'{self.Ki:.4f}' + 'iLim:' f'{self.iLim:.0f}' + 'Kaw:' f'{self.Kaw:.4f}')
            return -1

    def setSetTemp(self, Tset):
        self.oCtrl.SetPoint = Tset*10

    def setCalibration(self, Traw, Treal):
        self.factor = (Treal[1] - Treal[0]) / (Traw[1] - Traw[0])
        self.offset = Treal[0] - self.factor * Traw[0]
        print('offset: ' + f'{self.offset: .2f}' + ', factor: ' + f'{self.factor: .2f}')
        # Tcal = offset + Traw*factor
        self.oCtrl.setScale(Factor=self.factor, Offset=self.offset)

        self.offset = self.offset/10  # convert offset from 1/10deg in deg

    def getHeaterTemp(self):
        val = self.oCtrl.Value/10
        return val

    def getTipTemp(self):
        val = self.oAI.Value
        return val

    def getParam(self, par):
        nr = 0x00  # default value
        if par == 'pwm':  # getParam()
            nr = 0x05
        if par == 'Kp':  # getFloatParam()
            nr = 0x02
        if par == 'Ki':  # getFloatParam()
            nr = 0x03
        if par == 'u_i':  # getFloatParam()
            nr = 0x06
        if par == 'Kaw':  # getFloatParam()
            nr = 0x07
        if par == 'iLim':  # getFloatParam()
            nr = 0x08
        if par == 'u_p':  # getFloatParam()
            nr = 0x09

        if nr == 0x05:
            val = self.oCtrl.getParam(nr)
        elif par == 'Tset':  # read current SetPoint
            val = self.oCtrl.SetPoint/10
        else:
            val = self.oCtrl.getFloatParam(nr)

        return val

    def On(self):
        # set current temp to set-temp before power on
        curTemp = self.getHeaterTemp()
        self.setSetTemp(curTemp)
        self.oCtrl.cmdOn()

        # check if heater is really on
        isOn = self.oCtrl.isOn()
        if isOn == 1:
            if log:
                print('Heater ' + self.name + ' power on successful...')
            return 1
        else:
            if log:
                print('Heater ' + self.name + ' power on failed...')
            return -1

    def Off(self):
        self.oCtrl.cmdOff()

        # check if heater is really on
        isOn = self.oCtrl.isOn()
        if isOn == 1:
            if log:
                print('Heater ' + self.name + ' power off failed...')
            return -1
        else:
            if log:
                print('Heater ' + self.name + ' power off successful...')
            return 1


class DataCollector:
    # This class only collects data by reading values from the hardware.
    # The set-temperatures can be saved by the state-machine thread

    def __init__(self, ts, oThermode, oBHeater, ZAxis, oForce, oCool):
        self.Thread = threading.Thread(target=self.run)
        self.data = []
        self.ts = ts
        self.oThermode = oThermode
        self.oBHeater = oBHeater
        self.ZAxis = ZAxis
        self.oForce = oForce
        self.oCool = oCool
        self.init_data_dict()  # initialize data dictionary
        self.Stop = False
        now = datetime.now()
        sDateTime = now.strftime("%Y%m%d_%H%M%S")
        self.sDateTime = sDateTime

    def init_data_dict(self):
        # level 0
        data = {'Thermode': {}, 'BottomHeater': {}}
        # level 1
        data['Thermode'].update({'Kp': [], 'Ki': [], 'iLim': [], 'Kaw': []})
        data['Thermode'].update({'offset': [], 'factor': []})
        data['Thermode'].update({'u': [], 'u_p': [], 'u_i': []})
        data['Thermode'].update({'HeaterTemp': []})
        data['Thermode'].update({'SetTemp': []})
        data['Thermode'].update({'TipTemp': []})
        data['Thermode'].update({'ConstantPressure': []})
        data['Thermode'].update({'VariablePressure': []})
        data['Thermode'].update({'Airflow': []})
        data['Thermode'].update({'ZPosition': []})
        data['BottomHeater'].update({'Kp': [], 'Ki': [], 'iLim': [], 'Kaw': []})
        data['BottomHeater'].update({'offset': [], 'factor': []})
        data['BottomHeater'].update({'u': [], 'u_p': [], 'u_i': []})
        data['BottomHeater'].update({'HeaterTemp': []})
        data['BottomHeater'].update({'SetTemp': []})
        data['BottomHeater'].update({'Airflow': []})
        data['Time'] = []
        data['State'] = []
        data['dt_StateMachine'] = []
        data['dt_DataCollector'] = []

        self.data = data

    def start(self):
        self.Thread.start()

    def stop(self):
        self.Stop = True
        self.Thread.join()

    def run(self):
        tStart = time()
        _tOld = time()
        while not self.Stop:
            _tAbs = time()
            _tRel = _tAbs - tStart
            _dt = _tRel - _tOld
            _tOld = _tRel
            if not simMode:
                _thTempHeater = self.oThermode.getHeaterTemp()
                _thTempTip = self.oThermode.getTipTemp()
                _thTempSet = self.oThermode.getParam('Tset')
                _thU = self.oThermode.getParam('pwm')
                _thUp = self.oThermode.getParam('u_p')
                _thUi = self.oThermode.getParam('u_i')
                _zPos = ZAxis.ETLcmd('ML7')
                _bhTempHeater = self.oBHeater.getHeaterTemp()
                _bhTempSet = self.oBHeater.getParam('Tset')
                _bhU = self.oBHeater.getParam('pwm')
                _bhUp = self.oBHeater.getParam('u_p')
                _bhUi = self.oBHeater.getParam('u_i')
                _thFixPress, _thVarPress = self.oForce.getPressure()
                _thFixPress = _thFixPress._convert_magnitude('mbar')
                _thVarPress = _thVarPress._convert_magnitude('mbar')
                if self.oCool.isHigh():
                    _Airflow = 11.0
                else:
                    _Airflow = 0.0

            else:
                _thTempHeater = random.randint(90, 100)
                _thTempTip = random.randint(80, 90)
                _thTempSet = random.randint(0, 20)
                _thU = random.randint(60, 70)
                _thUp = random.randint(40, 50)
                _thUi = random.randint(55, 70)
                _zPos = random.randint(55, 70)
                _bhTempHeater = random.randint(95, 105)
                _bhTempSet = random.randint(15, 20)
                _bhU = random.randint(60, 70)
                _bhUp = random.randint(11, 15)
                _bhUi = random.randint(13, 18)
                _thFixPress = random.randint(4, 5)
                _thVarPress = random.randint(3, 7)
                _Airflow = 0.0

            self.data['Thermode']['HeaterTemp'] += [_thTempHeater]
            self.data['Thermode']['TipTemp'] += [_thTempTip]
            self.data['Thermode']['SetTemp'] += [_thTempSet]
            self.data['Thermode']['u'] += [_thU]
            self.data['Thermode']['u_p'] += [_thUp]
            self.data['Thermode']['u_i'] += [_thUi]
            self.data['Thermode']['ConstantPressure'] += [_thFixPress]
            self.data['Thermode']['VariablePressure'] += [_thVarPress]
            self.data['Thermode']['Airflow'] += [_Airflow]
            self.data['Thermode']['ZPosition'] += [_zPos]
            self.data['BottomHeater']['HeaterTemp'] += [_bhTempHeater]
            self.data['BottomHeater']['SetTemp'] += [_bhTempSet]
            self.data['BottomHeater']['u'] += [_bhU]
            self.data['BottomHeater']['u_p'] += [_bhUp]
            self.data['BottomHeater']['u_i'] += [_bhUi]
            self.data['BottomHeater']['Airflow'] += [_Airflow]
            self.data['dt_DataCollector'] += [_dt]
            self.data['Time'] += [_tAbs]
            sleep(self.ts - time() % self.ts)
        if not simMode:
            self.data['Thermode']['Kp'] = self.oThermode.Kp
            self.data['Thermode']['Ki'] = self.oThermode.Ki
            self.data['Thermode']['iLim'] = self.oThermode.iLim
            self.data['Thermode']['Kaw'] = self.oThermode.Kaw
            self.data['Thermode']['offset'] = self.oThermode.offset
            self.data['Thermode']['factor'] = self.oThermode.factor
            self.data['BottomHeater']['Kp'] = self.oBHeater.Kp
            self.data['BottomHeater']['Ki'] = self.oBHeater.Ki
            self.data['BottomHeater']['iLim'] = self.oBHeater.iLim
            self.data['BottomHeater']['Kaw'] = self.oBHeater.Kaw
            self.data['BottomHeater']['offset'] = self.oBHeater.offset
            self.data['BottomHeater']['factor'] = self.oBHeater.factor
        else:
            self.data['Thermode']['Kp'] = 0.0
            self.data['Thermode']['Ki'] = 0.0
            self.data['Thermode']['iLim'] = 0.0
            self.data['Thermode']['Kaw'] = 0.0
            self.data['Thermode']['offset'] = 0.0
            self.data['Thermode']['factor'] = 0.0
            self.data['BottomHeater']['Kp'] = 0.0
            self.data['BottomHeater']['Ki'] = 0.0
            self.data['BottomHeater']['iLim'] = 0.0
            self.data['BottomHeater']['Kaw'] = 0.0
            self.data['BottomHeater']['offset'] = 0.0
            self.data['BottomHeater']['factor'] = 0.0

    def write_to_json(self):
        # write data to json-file
        fileName = self.sDateTime + 'data' + '.txt'
        if simMode:
            fileName = 'sim_' + fileName
        with open(fileName, 'w') as outfile:
            json.dump(self.data, outfile)

    def plotTemp(self):
        # Plot data
        plt.figure(0)
        plt.plot(*zip(*self.data['Thermode']['HeaterTemp']), label='T_heater (thermode)')
        plt.plot(*zip(*self.data['Thermode']['TipTemp']), label='T_tip (thermode)')
        plt.plot(*zip(*self.data['Thermode']['SetTemp']), label='T_set (thermode)')
        plt.plot(*zip(*self.data['BottomHeater']['HeaterTemp']), label='T_heater (bottom-heater)')
        plt.plot(*zip(*self.data['BottomHeater']['SetTemp']), label='T_set (bottom-heater)')

        cmap = plt.get_cmap('tab20c')
        col = cmap.colors
        col += col
        for i in range(0, len(self.data['State']) - 1):
            color = col[i]
            tAxvspanStart = self.data['State'][i][1]
            tAxvspanEnd = self.data['State'][i + 1][1]
            plt.axvspan(tAxvspanStart, tAxvspanEnd, facecolor=color, alpha=0.4)
            labelState = self.data['State'][i][0] + ' (' + f'{tAxvspanStart:.1f}' + 's)'
            if i % 2 == 0:
                ymax = plt.gca().get_ylim()[1]
                plt.text(tAxvspanStart, ymax - 5, labelState)
            if i % 2 == 1:
                ymin = plt.gca().get_ylim()[0]
                plt.text(tAxvspanStart, ymin + 2, labelState)
        plt.legend()
        plt.xlabel('time in s')
        plt.ylabel('Temp. in deg')
        plt.grid()
        plt.tight_layout()

    def plotActuatingVars(self):
        # Plot data
        fig2 = plt.figure(1)
        plt.plot(*zip(*self.data['Thermode']['u']), label='u(t) (thermode)')
        plt.plot(*zip(*self.data['Thermode']['u_p']), label='u_p(t) (thermode)')
        plt.plot(*zip(*self.data['Thermode']['u_i']), label='u_i(t) (thermode)')
        plt.plot(*zip(*self.data['BottomHeater']['u']), label='u(t) (bottom-heater)')
        plt.plot(*zip(*self.data['BottomHeater']['u_p']), label='u_p(t) (bottom-heater)')
        plt.plot(*zip(*self.data['BottomHeater']['u_i']), label='u_i(t) (bottom-heater)')

        cmap = plt.get_cmap('tab20c')
        col = cmap.colors
        col += col
        for i in range(0, len(self.data['State']) - 1):
            color = col[i]
            tAxvspanStart = self.data['State'][i][1]
            tAxvspanEnd = self.data['State'][i + 1][1]
            plt.axvspan(tAxvspanStart, tAxvspanEnd, facecolor=color, alpha=0.4)
            labelState = self.data['State'][i][0] + ' (' + f'{tAxvspanStart:.1f}' + 's)'
            if i % 2 == 0:
                ymax = plt.gca().get_ylim()[1]
                plt.text(tAxvspanStart, ymax - 5, labelState)
            if i % 2 == 1:
                ymin = plt.gca().get_ylim()[0]
                plt.text(tAxvspanStart, ymin + 2, labelState)
        plt.legend()
        plt.xlabel('time in s')
        plt.ylabel('Control outputs in inc(0..1024)')
        plt.grid()
        plt.tight_layout()

    def saveFigs(self):
        plt.figure(0)
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        plt.figure(1)
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

        plt.figure(0)
        fileName = self.sDateTime + 'figTemp' + '.png'
        plt.savefig(fileName)
        plt.figure(1)
        fileName = self.sDateTime + 'figActVar' + '.png'
        plt.savefig(fileName)

        plt.show()


main()
'''oTh = Heater('HT_CT_02', 'HT_TEMP2_02')
oBh = Heater('HT_CT_00', 'HT_TEMP2_00')
oTh.setCalibration([1480, 1780], [1415, 1705])
valT = oTh.getHeaterTemp()
valB = oBh.getHeaterTemp()
print('TH: ' + str(valT))
print('BH: ' + str(valB))
oTh.On()
oTh.setSetTemp(148)
oCool = oma.DigitalOut('Cooling')
oCool.cmdSet()
oCool.cmdReset()
oTh.setCalibration([1480, 1780], [1432, 1732])
oBh.setCalibration([100, 200], [89, 189])
valT = oTh.getHeaterTemp()
valB = oBh.getHeaterTemp()
print('TH: ' + str(valT))
print('BH: ' + str(valB))
valT2 = oTh.getTipTemp()
valB2 = oBh.getTipTemp()
print('TH2: ' + str(valT2))
print('BH2: ' + str(valB2))

print()'''