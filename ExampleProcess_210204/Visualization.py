import json
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy import interpolate
import sys
from matplotlib.font_manager import FontProperties


class Visualization:

    def __init__(self, fname):
        self.fname = fname

        self.data = []
        self.dataStates = dict()
        self.read_from_json(fname)

        # sort data by states
        self.split_data_by_states()

        # set class colors
        self.colAxV = []
        self.colLine = []
        self.init_colors()

    def init_colors(self):
        cmap = plt.get_cmap('tab20c')
        col = cmap.colors
        col += col
        self.colAxV = col
        cmap = plt.get_cmap('tab10')
        col = cmap.colors
        col += col
        self.colLine = col

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

        return data

    def read_from_json(self, fname):
        with open(fname) as json_file:
            self.data = json.load(json_file)

    def split_data_by_states(self):
        vState = self.data['State']

        for i in range(0, len(vState) - 1):
            cState = vState[i][0]
            # check if state was entered more than once
            indices = [k for k in range(len(vState)) if
                       [x[0] for x in vState][k] == cState]  # find all appearances of the current state
            if len(indices) > 1:
                cPrevState = vState[i - 1][0]
                cState += '_after_' + cPrevState
            # search for index from i to end
            idx0State = [x[0] for x in vState[i:-1]].index(vState[i][0]) + i
            idx1State = idx0State + 1

            self.dataStates.update({cState: dict()})

            # get next state
            tStart = vState[idx0State][1]
            tEnd = vState[idx1State][1]

            time = self.data['Time']
            idx0Time = np.where(np.array(time) > tStart)[0][0]
            idx1Time = np.where(np.array(time) >= tEnd)[0]
            if len(idx1Time) == 0:
                idx1Time = len(time) - 1
            else:
                idx1Time = idx1Time[0]

            self.dataStates[cState] = self.trim_data_to_states(idx0Time, idx1Time, idx0State, idx1State)

    def trim_data_to_states(self, idx0Time, idx1Time, idx0State, idx1State):
        # extracts the data of a state by trimming the data vector and returning this data as a dict
        targetDict = self.init_data_dict()
        # iterate over top level keys (Thermode, BottomHeater, Time, ...)
        for topLvlKey in self.data:

            # iterate over nested dicts if they are of type dict
            if isinstance(self.data[topLvlKey], dict):
                for key in self.data[topLvlKey]:

                    if key not in targetDict[topLvlKey]:
                        targetDict[topLvlKey].update({key: []})
                    Vals = self.data[topLvlKey][key]
                    if isinstance(Vals, list):
                        targetDict[topLvlKey][key] = Vals[idx0Time:idx1Time]
                    elif isinstance(Vals, float):
                        targetDict[topLvlKey][key] = Vals
            elif topLvlKey == 'State':
                targetDict[topLvlKey] = self.data[topLvlKey][idx0State:idx1State] + [idx0State]

            elif isinstance(self.data[topLvlKey], list):
                Vals = self.data[topLvlKey]
                targetDict[topLvlKey] = Vals[idx0Time:idx1Time]
        return targetDict

    def get_state_dict(self, stateName):
        if stateName not in self.dataStates:
            sys.exit('Value: ' + stateName + ' is not a key of dictionary self.dataStates')

        return self.dataStates[stateName]

    def calc_ramp_rate(self, vTemp, vTime, vTempRef):
        # calc ramp rate between 10% and 90%
        T0 = vTempRef[0]
        T1 = vTempRef[-1]

        dT = T1 - T0

        T10 = T0 + dT * 0.1
        T90 = T1 - dT * 0.1

        idx10 = np.where(np.array(vTemp) >= T10)[0][0]
        idx90 = np.where(np.array(vTemp) >= T90)[0][0]

        t10 = vTime[idx10]
        t90 = vTime[idx90]
        T10 = vTemp[idx10]
        T90 = vTemp[idx90]

        rr = (T90 - T10) / (t90 - t10)

        return rr, t10, t90, T10, T90, T0, T1

    def plot_temp(self):
        time = self.data['Time']
        Th_th = self.data['Thermode']['HeaterTemp']
        Tt_th = self.data['Thermode']['TipTemp']
        Tset = self.data['Thermode']['SetTemp']
        trh = [Tset[x] - Th_th[x] for x in range(0, len(time))]
        trt = [Tset[x] - Tt_th[x] for x in range(0, len(time))]
        err_ht = [Th_th[x] - Tt_th[x] for x in range(0, len(time))]

        # resample and differentiate
        fcnTh_th = interpolate.interp1d(time, Th_th)
        time_rs = np.linspace(time[0], time[-1], len(time))
        Th_th_rs = fcnTh_th(time_rs[:])
        dTh_th = np.diff(Th_th_rs) / (time_rs[1] - time_rs[0])

        # Plot data
        plt.figure(0)
        plt.plot(time, Th_th, label='T_heater (thermode)')
        plt.plot(time, Tt_th, label='T_tip (thermode)')
        plt.plot(time, Tset, label='T_set (thermode)')
        plt.plot(time, self.data['BottomHeater']['HeaterTemp'], label='T_heater (bottom-heater)')
        plt.plot(time, self.data['BottomHeater']['SetTemp'], label='T_set (bottom-heater)')
        plt.plot(time, trh, label='Tset - T_heater (thermode)')
        plt.plot(time, trt, label='Tset - T_tip (thermode)')
        plt.plot(time, err_ht, label='T_heater - T_tip (thermode)')
        plt.plot(time_rs[1:], dTh_th, label='diff(T_heater) (thermode)')
        # plt.plot(time[1:], dTt_th, label='diff(T_tip) (thermode)')

        for i in range(0, len(self.data['State']) - 1):
            color = self.colAxV[i]
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

    def plot_actuatingVars(self):
        time = self.data['Time']
        # Plot data
        fig2 = plt.figure(1)
        plt.plot(time, self.data['Thermode']['u'], label='u(t) (thermode)')
        plt.plot(time, self.data['Thermode']['u_p'], label='u_p(t) (thermode)')
        plt.plot(time, self.data['Thermode']['u_i'], label='u_i(t) (thermode)')
        plt.plot(time, self.data['BottomHeater']['u'], label='u(t) (bottom-heater)')
        plt.plot(time, self.data['BottomHeater']['u_p'], label='u_p(t) (bottom-heater)')
        plt.plot(time, self.data['BottomHeater']['u_i'], label='u_i(t) (bottom-heater)')

        for i in range(0, len(self.data['State']) - 1):
            color = self.colAxV[i]
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

    def calc_ramp_properties(self, stateName):
        dictRamp = self.get_state_dict(stateName)
        dictHoldTemp = self.get_state_dict('HoldTemp_after_' + stateName)

        time = dictRamp['Time'] + dictHoldTemp['Time']
        vTempRef = dictRamp['Thermode']['SetTemp'] + dictHoldTemp['Thermode']['SetTemp']
        for i in ['Thermode', 'BottomHeater']:
            for k in ['SetTemp', 'HeaterTemp', 'TipTemp']:
                if i == 'BottomHeater' and k == 'TipTemp':
                    continue
                vTempSet = dictRamp[i][k] + dictHoldTemp[i][k]
                rr, t10, t90, T10, T90, T0, T1 = self.calc_ramp_rate(vTime=time, vTemp=vTempSet, vTempRef=vTempRef)
                self.dataStates[stateName][i].update({'Ramprate_' + k: dict()})
                self.dataStates[stateName][i]['Ramprate_' + k].update(
                    {'rr': rr, 't10': t10, 't90': t90, 'T10': T10, 'T90': T90, 'T0': T0, 'T1': T1})

    def plot_ramps(self, stateName):
        self.calc_ramp_properties(stateName)

        dictRamp = self.dataStates[stateName]
        dictHoldTemp = self.get_state_dict('HoldTemp_after_' + stateName)
        time = dictRamp['Time']
        timeHT = dictHoldTemp['Time']
        tOff = time[0]
        timeHT = [x - tOff for x in timeHT]
        time = [x - tOff for x in time]
        idxHT = np.where(timeHT >= np.array(time[-1])*1.2)[0][0]

        # Add a table at the bottom of the axes
        fig, ax = plt.subplots()
        fig.set_figheight(8.5)
        fig.set_figwidth(17)
        ax.plot(time+timeHT[0:idxHT], dictRamp['Thermode']['HeaterTemp']+dictHoldTemp['Thermode']['HeaterTemp'][0:idxHT], label='T_heater (thermode)', color=self.colLine[0])
        ax.plot(time+timeHT[0:idxHT], dictRamp['Thermode']['TipTemp']+dictHoldTemp['Thermode']['TipTemp'][0:idxHT], label='T_tip (thermode)', color=self.colLine[1])
        ax.plot(time+timeHT[0:idxHT], dictRamp['Thermode']['SetTemp']+dictHoldTemp['Thermode']['SetTemp'][0:idxHT], label='Set-Temp', color=self.colLine[2])
        ax.plot(time+timeHT[0:idxHT], dictRamp['BottomHeater']['HeaterTemp']+dictHoldTemp['BottomHeater']['HeaterTemp'][0:idxHT], label='T_heater (bottom-heater)', color=self.colLine[3])
        plt.axvspan(0, time[-1], facecolor=self.colAxV[dictRamp['State'][1]], alpha=0.4)
        plt.axvspan(time[-1], timeHT[idxHT], facecolor=self.colAxV[dictHoldTemp['State'][1]], alpha=0.4)
        ax.legend()
        ax.grid()
        ax.set_xlabel('time in s')
        ax.set_ylabel('Temperature in deg')

        # Create Table data
        tableData = []
        columns = ('Property', 'State', 'Unit', 'Value', 'Unit')
        for i in ['Thermode', 'BottomHeater']:
            for k in ['SetTemp', 'HeaterTemp', 'TipTemp']:

                if i == 'BottomHeater' and (k == 'TipTemp' or k == 'SetTemp'):
                    continue
                elif k == 'SetTemp':
                    val = self.dataStates[stateName][i]['Ramprate_' + k]['rr']
                    tableData.append(['Ramprate_' + k, stateName, 'Thermode,BottomHeater', f'{val:.2f}', 'deg/s'])
                else:
                    val = self.dataStates[stateName][i]['Ramprate_' + k]['rr']
                    tableData.append(['Ramprate_' + k, stateName, i, f'{val:.2f}', 'deg/s'])

                # set correct color
                if k == 'SetTemp':
                    cl = self.colLine[2]
                elif k == 'HeaterTemp':
                    if i == 'Thermode':
                        cl = self.colLine[0]
                    elif i == 'BottomHeater':
                        cl = self.colLine[3]
                elif k == 'TipTemp':
                    cl = self.colLine[1]
                # plot 10% and 90% data points
                t10 = self.dataStates[stateName][i]['Ramprate_' + k]['t10'] - tOff
                t90 = self.dataStates[stateName][i]['Ramprate_' + k]['t90'] - tOff
                T10 = self.dataStates[stateName][i]['Ramprate_' + k]['T10']
                T90 = self.dataStates[stateName][i]['Ramprate_' + k]['T90']
                plt.scatter([t10, t90], [T10, T90], color=cl)

        tableData.append(['Time of theoretical ramp', stateName, 'Thermode,BottomHeater', f'{time[-1]:.2f}', 's'])
        T0 = self.dataStates[stateName]['Thermode']['Ramprate_SetTemp']['T0']
        T1 = self.dataStates[stateName]['Thermode']['Ramprate_SetTemp']['T1']
        tableData.append(['Start Temperature', stateName, 'Thermode,BottomHeater', f'{T0:.2f}', 'deg'])
        tableData.append(['End Temperature', stateName, 'Thermode,BottomHeater', f'{T1:.2f}', 'deg'])

        rcolors = np.full(len(tableData[::-1]), '#D18F77')
        the_table = ax.table(cellText=tableData,
                             colLabels=columns,
                             cellLoc='center',
                             rowLoc='center',
                             colColours=rcolors,
                             loc='bottom',
                             bbox=[0.0, -0.4, 1.0, 0.3]
                             )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        for (row, col), cell in the_table.get_celld().items():
            if (row == 0) or (col == -1):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        plt.subplots_adjust(left=0.05, bottom=0.3)

        # Adjust layout to make room for the table:
        plt.title('Analysis of state ' + stateName + ' in file ' + self.fname)


fname = '20210209_141427data.txt'
vis = Visualization(fname)

# plot data
vis.plot_temp()
# vis.plot_actuatingVars()
vis.plot_ramps('Ramp1')
vis.plot_ramps('Ramp2')
plt.show()
