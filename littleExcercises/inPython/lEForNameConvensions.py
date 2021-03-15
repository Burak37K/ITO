import random

#RSP @BurakHotan

def getAllRandomRSPValues():

    playerAValues = []
    playerBValues = []

    EndValueOfLoop = 10
    for StartValueOfLoop in range(EndValueOfLoop):
        playerARandomGeneratedValues = random.randrange(0,3)
        playerBRandomGeneratedValues = random.randrange(0,3)
        playerAValues.append(playerARandomGeneratedValues)
        playerBValues.append(playerBRandomGeneratedValues)
    return playerAValues,playerBValues


def compareValuesOfBothPlayers(VariablesOfRandomGeneratedRSPValues):

    CollectMatchResults = []
    ArrayIndexOfPlayerA = 0
    ArrayIndexOfPlayerB = 1
    EndValueOfLoop = 10
    for StartValueOfLoop in range(EndValueOfLoop):
        if VariablesOfRandomGeneratedRSPValues[ArrayIndexOfPlayerA][StartValueOfLoop] > VariablesOfRandomGeneratedRSPValues [ArrayIndexOfPlayerB] [StartValueOfLoop] :
            CollectMatchResults.append(('Winner A'))
        elif  VariablesOfRandomGeneratedRSPValues[ArrayIndexOfPlayerA][StartValueOfLoop] < VariablesOfRandomGeneratedRSPValues [ArrayIndexOfPlayerB] [StartValueOfLoop]:
            CollectMatchResults.append(('Winner B'))
        elif  VariablesOfRandomGeneratedRSPValues[ArrayIndexOfPlayerA][StartValueOfLoop] == VariablesOfRandomGeneratedRSPValues [ArrayIndexOfPlayerB] [StartValueOfLoop]:
            CollectMatchResults.append(('draw'))
    return CollectMatchResults


def emittedTheWinnerFunction(ResultsofRPCMatches):

    EndValueOfLoop = 10

    for StartLoopIndex in range(EndValueOfLoop):
        PointsofPlayerA = ResultsofRPCMatches.count('Winner A')
        PointsofPlayerB = ResultsofRPCMatches.count('Winner B')
        if PointsofPlayerA > PointsofPlayerB:
            return 'Winner A'
        else:
            return 'Winner B'

def intoMainPorgamm():

    #3 Funktionen
    #1: Funktion: Generieren von Werten RSP
    #2: Vergleichen von Ergebnissen
    #3: Zusammenzählen

    VariablesOfRandomGeneratedRSPValues =  getAllRandomRSPValues()
    ResultsofRPCMatches = compareValuesOfBothPlayers(VariablesOfRandomGeneratedRSPValues)
    EmittedTheWinnerValue = emittedTheWinnerFunction(ResultsofRPCMatches)
    print(EmittedTheWinnerValue)

intoMainPorgamm()

