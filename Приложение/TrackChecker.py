import PySimpleGUI as sg
import matplotlib.pyplot as plt
import predict as prd
import warnings
import pandas as pd
import ex

warnings.filterwarnings("ignore")


def to_seconds(x):
    h, m, s = x.split(':')
    result = int(h) * 60 * 60 + int(m) * 60 + int(s)
    return result


good = []
bad = []


def plot(tmp):
    plt.figure(figsize=(8, 7))

    plt.subplot(2, 2, 1)
    plt.scatter(tmp.Latitude.values, tmp.Longitude.values, s=2)
    plt.title("Latitude_Longitude")

    plt.subplot(2, 2, 2)
    plt.scatter(tmp.Time.values, tmp.Longitude.values, s=2)
    plt.title("Time_Longitude")

    plt.subplot(2, 2, 3)
    plt.scatter(tmp.Time.values, tmp.Latitude.values, s=2)
    plt.title("Time_Latitude")

    plt.subplot(2, 2, 4)
    plt.scatter(tmp.Time.values, tmp.Height.values, s=2)
    plt.title("Time_Height")

    plt.show(block=False)


layout = [
    [sg.Text('File for check'), sg.InputText(), sg.FileBrowse(),
     ],
    [sg.Output(size=(78, 20))],
    [sg.Submit(), sg.Button('Put in files')],
    [sg.Text('Enter id'), sg.InputText(), sg.Button('Show trace')],
    [sg.Exit()]
]
window = sg.Window('Track Checker', layout)
while True:                             # The Event Loop
    event, values = window.read()
    # print(event, values) #debug
    if event in (None, 'Exit', 'Cancel'):
        break
    elif event == 'Submit':
        if values[0] == '':
            print("Please, select file.")
        else:
            good, bad, frame_of_uns = prd.PREDICT(values[0])

            frame_of_uns.to_csv('id_to_class_probabilities.txt', sep=' ', index=False)

            print("In file: ", values[0])
            print("--------------------")
            print("good tracks id")
            print("--------------------")
            for i in range(len(good)):
                print(good[i], end=" ")
            print("\n")
            print("--------------------")
            print("bad tracks id")
            print("--------------------")
            for i in range(len(bad)):
                print(bad[i], end=" ")
            print("\n")
            print('id_to_class_probabilities.txt created!')

    elif event == 'Show trace':
        if values[1] == '':
            print("Please, enter correct ID.")
        else:
            df = pd.read_csv(values[0], sep=" ", header=None)
            columns = ['Time', 'ID', 'Latitude', 'Longitude', 'Height', 'Code', 'Name']
            df.columns = columns
            tmp = df[df.ID == int(values[1])].copy()
            tmp.Time = tmp.Time.apply(to_seconds)
            plot(tmp)

    elif event == 'Put in files':
        if len(good) == 0 and len(bad) == 0:
            print("Firstly, submit file.")
        else:
            ex.put_file(values[0], good, bad)
            print("BadTrack.txt and GoodTrack.txt created!")