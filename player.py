from hand import Hand
from typing import List
from bet import spread1_50, spread1_6
class Player:
    def __init__(self, name, initialBankroll, strategy, betting, isVerbose):
        if isVerbose: print("Creating new player: ", name)
        self.name = name
        self.bankroll = initialBankroll
        self.bankrollSnapshots = [initialBankroll]
        self.strategy = strategy
        self.betting = betting
        self.isVerbose = isVerbose
        self.hands: List[Hand] = []
        self.handData = [0, 0, 0] # [Wins, Losses, Draws]
        self.reward = 0
    
    def calculateBetSize(self, tableMin, trueCount):
        if self.strategy.isCounting:
            return self.betting.getBetSpreads(trueCount, tableMin)
        return tableMin
    
    def canPlay(self):
        return len(self.hands) > 0
    
    def clearHand(self, hand: Hand):
        self.hands.remove(hand)
    
    def clearAllHands(self):
        self.hands.clear()
    
    def getStartingHand(self):
        if len(self.hands) == 0:
            raise ValueError(f"Error: el jugador {self.name} no tiene manos asignadas.")
        return self.hands[0]


    
    def splitPair(self, hand: Hand):
        splitHand = Hand([hand.splitHand()], hand.getInitialBet())
        self.updateHand(splitHand)
        return splitHand
    
    def takeBankrollSnapshot(self):
        self.bankrollSnapshots.append(self.bankroll)
        bankrollDiff = self.bankrollSnapshots[len(self.bankrollSnapshots) - 1] - self.bankrollSnapshots[len(self.bankrollSnapshots) - 2]
        if bankrollDiff > 0:
            self.handData[0] += 1
        elif bankrollDiff < 0:
            self.handData[1] += 1
        else:
            self.handData[2] += 1
    
    def updateBankroll(self, amount):
        self.bankroll = self.bankroll + amount
    
    def updateHand(self, hand):
        self.hands.append(hand)
