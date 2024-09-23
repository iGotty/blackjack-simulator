from shoe import Shoe
from card import Card
from hand import Hand
from typing import List
from count import HiLoCount

class HouseRules:
    def __init__(self, standValue, DASoffered, RSAoffered, LSoffered, doubleOnSoftTotal):
        self.standValue = standValue
        self.doubleAfterSplitOffered = DASoffered
        self.resplitAcesOffered = RSAoffered
        self.lateSurrenderOffered = LSoffered
        self.doubleOnSoftTotal = doubleOnSoftTotal
        print("HOUSE RULES:")
        print("Dealer stands on ", standValue, " | Double after split offered? ", DASoffered, " | Players can re-split aces? ", RSAoffered, " Surrender offered? ", LSoffered)
class Dealer:
    def __init__(self, penetration, shoeSize, houseRules, strategy, isVerbose):
        self.penetration = penetration
        self.shoe = Shoe(shoeSize, isVerbose)
        self.houseRules: HouseRules = houseRules
        self.strategy = strategy
        self.isVerbose = isVerbose
        self.hand: Hand = None
        self.upcard: Card = None
        self.losses = 0
        self.gains = 0
        self.count = HiLoCount()
    
    def dealCard(self):
        return self.shoe.drawCard()
    
    def discardDealersCards(self):
        if self.isVerbose: print("Discarding dealer hand: ", self.hand.cards)
        for card in self.hand.cards:
            self.shoe.discardCard(card)
        self.hand = None
        self.upcard = None

    def discardPlayersCards(self, hand: Hand, playerName):
        if self.isVerbose: print("Discarding ", playerName, "'s hand...")
        for card in hand.cards:
            self.shoe.discardCard(card)
    
    def deckPenetrationTooHigh(self):
        return self.shoe.getPenetration() >= self.penetration
    
    def ensureDeckCompleteness(self, isVerbose):
        if (len(self.shoe.discard) + len(self.shoe.drawPile) != self.shoe.numDecks * 52):
            if self.isVerbose: 
                print("Length of discard and draw piles: ", len(self.shoe.discard), " + ", len(self.shoe.drawPile), " = ", len(self.shoe.discard)+len(self.shoe.drawPile))
                print("Should be equal to: ", self.shoe.numDecks * 52)
            raise SystemExit('ERROR: There are cards missing! ')
        return True

    def handlePayout(self, betSize: int, isBlackjack):
        if isBlackjack:
            self.losses += betSize * 1.5
            return betSize * 1.5
        self.losses += betSize
        return betSize
    
    def insuranceIsOffered(self):
        return self.upcard.getValue() == 11

    def setUpCard(self, upcard: Card):
        self.upcard = upcard
    
    def shuffle(self):
        self.shoe.resetShoe()
    
    def updateHand(self, hand: Hand):
        self.hand = hand
    
    def updateGains(self, amount):
        self.gains += amount