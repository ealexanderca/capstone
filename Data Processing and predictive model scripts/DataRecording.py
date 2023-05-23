# MECH 457 Fujitsu Capstone Data Logging system
# Ethan Alexander 84207034
# This program automatically logs data from selected inputs on a T7 daq and stores it into a sqlite database
# make sure to disable all microphones except for the one used to record to ensure it records from the correct source
# all configuration can be done in the settings below
# this program is based on examples from the LabJack LJM GITHUB
import sys
import sounddevice as sd
import queue
import sqlite3
import re
import datetime
from labjack import ljm
import keyboard
from statistics import mean
import numpy as np

# queue that stores sound data for processing
q = queue.Queue()


def callback(indata, frames, time, status):  # defines function that enters sound data into the queue
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


# settings
DBName = 'heatertest'  # compressor name
dutyCycle = .5  #controls the on off time of the acceleration
processFreq = 2  # frequency to process data
logSec = 10  # float('inf')  # active logging time set to infinity to continuously record. note when recording infinitely it will only record the slow channel once at the start
gapSec = 0  # time between starts of recording intervals set less than logSec to record as soon as possible
soundFrequency = 48000  # recording frequency of the sound
dataScanRate = 100000  # total number of values read per second max 100 000 but must be set lower if not reading +/-10V on all fast channels
maxDataFrequency = 50000  # will limit the per channel frequency if it is lower than the data scan rate/number of channels
fastScan = ["AIN0"] + ["AIN2"] + ["AIN4"] + ["DIO6_EF_READ_A"] + ["DIO3_EF_READ_A"] # values that are scanned at full frequency
slowScan = ["AIN6"] + ["AIN8_EF_READ_A"] + ["AIN12_EF_READ_A"] # values that are scanned before every fast interval
slowSamples = 100  # number of samples for the slow scans to average

# CODE

if logSec == float("inf"):  # if the recording period is set to infinite it will not record slow data
    slowScan = []
fastDataTableStr = " DOUBLE, ".join(fastScan) + " DOUBLE"
slowDataTableStr = " DOUBLE, ".join(slowScan) + " DOUBLE"
# finds the number of channels being recorded
fastCh = len(fastScan)
slowCh = len(slowScan)
dataFrequency = min([dataScanRate / fastCh, maxDataFrequency])
fastDataExecStr = "?, ?"
for i in range(fastCh):
    fastDataExecStr += ", ?"
slowDataExecStr = "?, ?, ?"
for i in range(slowCh):
    slowDataExecStr += ", ?"
ljm.closeAll()
handle = ljm.openS("T7", "ANY", "ANY")  # T7 device, Any connection, Any identifier
print(DBName)
con = sqlite3.connect(DBName+'.db')
cur = con.cursor()
res = cur.execute("SELECT name FROM sqlite_master")
tabList = str(res.fetchall())
if len(tabList) == 2:
    con.execute("CREATE TABLE HEADER(TEST integer,TIME DATETIME , DESCRIPTION TEXT, DATA_FREQUENCY DOUBLE, SOUND_FREQUENCY DOUBLE)")
num = re.findall(r'\d+', tabList)
if num:
    num = max([int(s) for s in num]) + 1
else:
    num = 1

info = ljm.getHandleInfo(handle)
print("Opened a LabJack with Device type: %i, Connection type: %i,\n"
      "Serial number: %i, IP address: %s, Port: %i,\nMax bytes per MB: %i" %
      (info[0], info[1], info[2], ljm.numberToIP(info[3]), info[4], info[5]))
deviceType = info[0]

# DAC configuration
aNames = ["AIN_ALL_NEGATIVE_CH", "AIN_ALL_RANGE", "STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"]
aValues = [ljm.constants.GND, 10.0, 0, 0]  # single-ended, +/-10V, 0 (default), 0 (default)
ljm.eWriteNames(handle, len(aNames), aNames, aValues)
# current sensor
# log "AIN0"
# Vs = White Wire
# GND = Green Wire + Resistor Leg
# AIN0 = Red wire + Resistor Leg
# AIN1 = Black Wire
ljm.eWriteName(handle, "AIN0_NEGATIVE_CH", 1)
ljm.eWriteName(handle, "AIN0_RANGE", 10)
# voltage sensor
# log "AIN2"
# AIN3 = White Wire
# AIN2 = Green Wire
ljm.eWriteName(handle, "AIN2_NEGATIVE_CH", 3)
ljm.eWriteName(handle, "AIN2_RANGE", 10)
# first pressure sensor
# log "AIN4"
# using LJTIA set to a gain of 61 and an offset of 0.4V
# Vref = Orange Wire
# GND = Orange White Wire
# INA+ = Green Wire
# INA- = Green White Wire
ljm.eWriteName(handle, "AIN4_NEGATIVE_CH", 5)
ljm.eWriteName(handle, "AIN4_RANGE", 10)
# second pressure sensor
# log"AIN6"
# using LJTIA set to a gain of 61 and an offset of 0.4V
# Vref = Orange Wire
# GND = Orange White Wire
# INA+ = Green Wire
# INA- = Green White Wire
ljm.eWriteName(handle, "AIN6_NEGATIVE_CH", 7)
ljm.eWriteName(handle, "AIN6_RANGE", 10)
# thermocouple 1
# log "AIN8_EF_READ_A"
# AIN8 = Yellow Wire
# AIN9 = Red Wire
ljm.eWriteName(handle, "AIN8_NEGATIVE_CH", 9)
ljm.eWriteName(handle, "AIN8_RANGE", .1)
ljm.eWriteName(handle, "AIN8_EF_INDEX", 22)
ljm.eWriteName(handle, "AIN8_EF_CONFIG_A", 0)
ljm.eWriteName(handle, "AIN8_EF_CONFIG_B", 60052)
ljm.eWriteName(handle, "AIN8_EF_CONFIG_C", 0)
ljm.eWriteName(handle, "AIN8_EF_CONFIG_D", 1)
# thermocouple 2
# log "AIN10_EF_READ_A"
# AIN10 = Yellow Wire
# AIN11 = Red Wire
ljm.eWriteName(handle, "AIN10_NEGATIVE_CH", 11)
ljm.eWriteName(handle, "AIN10_RANGE", .1)
ljm.eWriteName(handle, "AIN10_EF_INDEX", 22)
ljm.eWriteName(handle, "AIN10_EF_CONFIG_A", 0)
ljm.eWriteName(handle, "AIN10_EF_CONFIG_B", 60052)
ljm.eWriteName(handle, "AIN10_EF_CONFIG_C", 0)
ljm.eWriteName(handle, "AIN10_EF_CONFIG_D", 1)
# spare channel
ljm.eWriteName(handle, "AIN12_NEGATIVE_CH", 13)
ljm.eWriteName(handle, "AIN12_RANGE", .1)
ljm.eWriteName(handle, "AIN12_EF_INDEX", 22)
ljm.eWriteName(handle, "AIN12_EF_CONFIG_A", 0)
ljm.eWriteName(handle, "AIN12_EF_CONFIG_B", 60052)
ljm.eWriteName(handle, "AIN12_EF_CONFIG_C", 0)
ljm.eWriteName(handle, "AIN12_EF_CONFIG_D", 1)
# rotation index counter
# log "DIO0_EF_READ_A"
# FIO5 = Index Wire
ljm.eWriteName(handle, "DIO3_EF_ENABLE", 0)
ljm.eWriteName(handle, "DIO3_EF_INDEX", 9)
ljm.eWriteName(handle, "DIO3_EF_ENABLE", 1)
# digital counter for quadrature
# enable speed boolean or log "DIO2_EF_READ_A"
# FIO6 = QUAD A
# FIO7 = QUAD B
ljm.eWriteName(handle, "DIO6_EF_ENABLE", 0)
ljm.eWriteName(handle, "DIO7_EF_ENABLE", 0)
ljm.eWriteName(handle, "DIO6_EF_INDEX", 10)
ljm.eWriteName(handle, "DIO7_EF_INDEX", 10)
ljm.eWriteName(handle, "DIO6_EF_ENABLE", 1)
ljm.eWriteName(handle, "DIO7_EF_ENABLE", 1)
# high speed counters for quad
ljm.eWriteName(handle, "DIO_EF_CLOCK0_ENABLE", 0)
ljm.eWriteName(handle, "DIO_EF_CLOCK1_ENABLE", 0)
ljm.eWriteName(handle, "DIO_EF_CLOCK2_ENABLE", 0)
ljm.eWriteName(handle, "DIO16_EF_ENABLE", 0)
ljm.eWriteName(handle, "DIO17_EF_ENABLE", 0)
ljm.eWriteName(handle, "DIO16_EF_INDEX", 7)
ljm.eWriteName(handle, "DIO17_EF_INDEX", 7)
ljm.eWriteName(handle, "DIO16_EF_ENABLE", 1)
ljm.eWriteName(handle, "DIO17_EF_ENABLE", 1)
# high speed counter for index
ljm.eWriteName(handle, "DIO18_EF_ENABLE", 0)
ljm.eWriteName(handle, "DIO18_EF_INDEX", 7)
ljm.eWriteName(handle, "DIO18_EF_ENABLE", 1)
# Ensure triggered stream is disabled.
ljm.eWriteName(handle, "STREAM_TRIGGER_INDEX", 0)
# Enabling internally-clocked stream.
ljm.eWriteName(handle, "STREAM_CLOCK_SOURCE", 0)
# Increases the buffer size to ensure no lost data
ljm.eWriteName(handle, "STREAM_BUFFER_SIZE_BYTES", 32768)

aScanList = ljm.namesToAddresses(fastCh, fastScan)[0]
scansPerRead = dataFrequency / processFreq
soundIn = sd.InputStream(samplerate=soundFrequency, dtype='float32', channels=1, callback=callback)
desc = input('enter a description of test:\n')
cur.execute("INSERT INTO HEADER VALUES(?, ?, ?, ?, ?)", (num, datetime.datetime.now(), desc, dataFrequency, soundFrequency))
# create tables for each frequency
con.execute("CREATE TABLE SOUND" + str(num) + "(SECTION INT, TIME DATETIME , SOUND DOUBLE)")
con.execute("CREATE TABLE FASTDATA" + str(num) + "(SECTION INT, TIME DATETIME , " + fastDataTableStr + ")")
con.execute("CREATE TABLE SLOWDATA" + str(num) + "(SECTION INT, TIME DATETIME , ACCELERATE INT,  " + slowDataTableStr + ")")
# Configure and start stream

# start of loop section
section = 0
acc = 0
while True:
    try:
        # reset values
        data = []
        totScans = 0
        totSkip = 0
        totFrames = 0
        # slow data sampling
        print("Reading slow scan Data")
        start = datetime.datetime.now()
        if slowCh > 0:
            slowData = [[] for _ in range(slowCh)]
            slowMean = []
            for _ in range(slowSamples):
                temp = (ljm.eReadNames(handle, slowCh, slowScan))
                for i in range(len(temp)):
                    if temp[i] == -9999.0:
                        slowData[i].append(np.nan)
                    else:
                        slowData[i].append(temp[i])
            for i in range(slowCh):
                if slowData[i].count(np.nan) == slowSamples:
                    slowMean += [np.nan]
                else:
                    slowMean += [np.nanmean(slowData[i])]
            data = (section, start, acc)
            for i in range(slowCh):
                data += (slowMean[i],)
            cur.execute("INSERT INTO SLOWDATA" + str(num) + " VALUES(" + slowDataExecStr + ")", data)
            con.commit()
            ainStr = ""
            for i in range(0, slowCh):
                ainStr += "%s = %0.5f, " % (slowScan[i], slowMean[i])
            print("  Slow Read Data: %s" % ainStr)
            if keyboard.is_pressed('q'):
                break
        # fast data sampling
        start2 = datetime.datetime.now()
        dataFrequency = ljm.eStreamStart(handle, int(scansPerRead), fastCh, aScanList, dataFrequency)
        sd.InputStream.start(soundIn)
        print("\nStream started with a scan rate of %0.0f Hz." % dataFrequency)
        i = 0
        if acc==1:
            ljm.eWriteName(handle,"DAC1",5)
            # start acceleration method
        while i < logSec * processFreq:
            if i <= int(dutyCycle * logSec * processFreq) and acc == 1:
                ljm.eWriteName(handle, "DAC1", 5)
            else:
                ljm.eWriteName(handle, "DAC1", 0)
            ret = ljm.eStreamRead(handle)
            curData = ret[0]
            scans = int(len(curData) / fastCh)
            curSkip = curData.count(-9999.0)
            #curData[curData == -9999] = np.nan
            totSkip += curSkip
            data = (np.ones(scans)*section, np.arange(start2 + datetime.timedelta(seconds=totScans/dataFrequency), start2 + datetime.timedelta(seconds=(totScans+scans)/dataFrequency), datetime.timedelta(seconds=1/dataFrequency)))
            for j in range(fastCh):
                data += (curData[j::fastCh],)
            data = np.transpose(data)
            cur.executemany("INSERT INTO FASTDATA" + str(num) + " VALUES(" + fastDataExecStr + ")", data)
            qLen = q.qsize()
            sound = []
            for j in range(qLen):
                sound += q.get().tolist()
            frames = len(sound)
            if frames > 0:
                data = zip(np.ones(frames)*section, np.arange(start2 + datetime.timedelta(seconds=totFrames/soundFrequency), start2 + datetime.timedelta(seconds=(totFrames+frames)/soundFrequency), datetime.timedelta(seconds=1/soundFrequency)), sound[:][0])
                cur.executemany("INSERT INTO SOUND" + str(num) + " VALUES(?, ?, ?)", tuple(data))
                con.commit()
            totFrames += frames
            totScans += scans
            i += 1
            if i % processFreq == 0:
                print("\neStreamRead %i" % i)
                ainStr = ""
                for j in range(0, fastCh):
                    ainStr += "%s = %0.5f, " % (fastScan[j], curData[j])
                print("  1st  out of %i: %s" % (scans, ainStr))
                print("  totScans Skipped = %0.0f, Scans Skipped = %0.0f, Scan Backlogs: Device = %i, LJM = " "%i" % (totSkip / fastCh, curSkip / fastCh, ret[1], ret[2]))
                print("  Sound queue length = %0.0f" % qLen)
                print("  acceleration = %0i" % acc )
            if keyboard.is_pressed('q') or keyboard.is_pressed('w'):
                break
        ljm.eWriteName(handle, "DAC1", 0)
        print("\nStop Stream")
        end = datetime.datetime.now()
        sd.InputStream.stop(soundIn)
        ljm.eStreamStop(handle)
        tt = (end - start2).seconds + float((end - start2).microseconds) / 1000000
        print("\nTotal scans = %i" % totScans)
        print("Time taken = %f seconds" % tt)
        print("LJM Scan Rate = %f scans/second" % dataFrequency)
        print("Timed Scan Rate = %f scans/second" % (totScans / tt))
        print("Timed Sample Rate = %f samples/second" % (totScans * fastCh / tt))
        print("Skipped scans = %0.0f" % (totSkip / fastCh))
        print("Sound file length = %f" % (totFrames / soundFrequency))
        section += 1
        while gapSec > (datetime.datetime.now() - start).seconds:
            if keyboard.is_pressed('q'):
                break
        if keyboard.is_pressed('w'):
            if acc == 0:
                acc = 1
            else:
                acc = 0
        if keyboard.is_pressed('q'):
            break
    except ljm.LJMError:
        ljme = sys.exc_info()[1]
        print(ljme)
        ljm.close(handle)
        handle = ljm.openS("T7", "ANY", "ANY")
    except Exception:
        e = sys.exc_info()[1]
        print(e)
        ljm.close(handle)
        handle = ljm.openS("T7", "ANY", "ANY")
    if keyboard.is_pressed('q'):
        break
ljm.close(handle)
# Close handle
