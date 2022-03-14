from appJar import gui
import numpy as np
import random
import matplotlib.pyplot

def press(btn):
        print(btn)

def getXY():
    x = np.arange(0.0, 3.0, 0.01)
    y = np.sin(random.randint(1,10) * np.pi * x)
    return x,y 

def generate(btn):
    # *getXY() will unpack the two return values
    # and pass them as separate parameters
    app.updatePlot("p1", *getXY())
    showLabels()

def showLabels():
    axes.legend(['The curve'])
    axes.set_xlabel("X Axes")
    axes.set_ylabel("Y Axes")
    app.refreshPlot("p1")

# start the GUI
app = gui("ML Final Project", " 800x900")
# add & configure widgets - widgets get a name, to help referencing them later
app.setFont(20)
app.setSticky("new")
app.setStretch("both")
#Make scrollable window

app.setStretch("both")
app.addLabel("l0", "ML interface")
app.setLabelBg("l0", "blue")


app.setSticky("new")
#app.addLabel("l2", "Options")
app.setStretch("both")
app.addLabelOptionBox("Select a Crypto",["   - Cryptocurrencies -", "Bitcoin", "Ethereum",
                        "Tether", "ADA", "DOGE", "AVAX", "XTZ",
                        "SHIB", "DOT","SOL"])
#app.setStretch("both")
app.startScrollPane("PANE")
app.setStretch("both")
app.startLabelFrame("Timeframe for prediction")

#app.setFont(14)
#app.setSticky("ew")
app.setFont(20)
app.addLabelEntry("Starting Day:    ")
app.addLabelEntry("Starting Month: ")
app.addLabelEntry("Starting Year:   ")
StartingDay = app.getEntry("Starting Day:    ")
StartingMonth = app.getEntry("Starting Month: ")
StartingYear = app.getEntry("Starting Year:   ")

app.setFont(14)
app.addLabel("a", " ")
app.addLabelEntry("Ending Day:    ")
app.addLabelEntry("Ending Month: ")
app.addLabelEntry("Ending Year:   ")
EndingDay = app.getEntry("Ending Day:    ")
EndingMonth = app.getEntry("Ending Month: ")
EndingYear = app.getEntry("Ending Year:   ")
app.stopLabelFrame()


app.setStretch("both")
app.addButton("Predict", press)
app.setStretch("both")
app.setFont(20)
app.addLabelOptionBox("Trading Interval", ["1 Hour", "12 hour",
"Day", "Week","Month","Year", "5 Year"])
#Plot
axes = app.addPlot("p1", *getXY())
showLabels()
app.setSticky("ew")
app.addButton("Generate", generate)

app.stopScrollPane()
app.go()
