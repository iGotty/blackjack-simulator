import click
from shoe import Shoe
from player import Player
from count import HiLoCount
from hand import Hand
from card import Card, CardValue
from dealer import Dealer, HouseRules
from strategies import BasicStrategy, CasinoStrategy, RandomStrategy, GameActions
from bet import spread1_50, spread1_6
from typing import List
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

@click.command()
@click.option('-s', '--shoesize', default=6, help='An integer representing the amount of decks to use in the shoe. Default is 6 decks.')
@click.option('-b', '--bankroll', default=10000, help='Determines the amount of dollars that each player begins the game with in their bankroll. Default is $1000.')
@click.option('-h', '--hands', default=1000, help='Determines the number of hands to deal. Default is 1000.')
@click.option('-t', '--tablemin', default=10, help='Determines the minimum table bet. Default is 5.')
@click.option('-p', '--penetration', default=0.84, help='Dictates the deck penetration by the dealer. Default is 0.84 which means that the dealer will penetrate 84 percent of the shoe before re-shuffling')
@click.option('-d', '--dealersettings', default=[17, True, True, True, True], help='Assigns the dealer rules.')
@click.option('-v', '--verbose', default=False, help='Prints all player, hand, and game information.')
def main(shoesize, bankroll, hands, tablemin, penetration, dealersettings, verbose):
    print("Running blackjack simulation with variables:")
    print("Shoe size: ", shoesize, " | Bankroll: ", bankroll, " | Number of hands to simulate: ", hands, " | Minimum Table Bet: ", tablemin)
    houseRules = HouseRules(standValue=dealersettings[0], DASoffered=dealersettings[1], RSAoffered=dealersettings[2], LSoffered=dealersettings[3], doubleOnSoftTotal=dealersettings[4])
    game = BlackJackGame(shoesize, bankroll, hands, tablemin, penetration, houseRules)
    game.startGame(isVerbose=True)
    gamedata = GameData(game)
    gamedata.getDealerStatistics()
    gamedata.plotBankrollTime()

class GameData:
    def __init__(self, game):
        self.game: BlackJackGame = game
        self.players = self.game.players
        self.dealer = self.game.dealer
        self.bankrollData = {}
        self.getPlayerBankrollSnapshots()
    
    def getPlayerBankrollSnapshots(self):
        for player in self.players:
            self.bankrollData.update({player.name: player.bankrollSnapshots})
    
    def getDealerStatistics(self):
        print(" - - - - - ")
        print("Dealer losses: $", self.dealer.losses, " | Dealer gains: $", self.dealer.gains, " | Profit: $", self.dealer.gains - self.dealer.losses)
        print(" - - - - - ")
    
    def plotBankrollTime(self):
        numHands = self.game.numHands
        playerNames = []
        fig, ax = plt.subplots()
        for player in self.players:
            ax.plot([item for item in range(1, len(player.bankrollSnapshots) + 1)], player.bankrollSnapshots, label=player.name)
            print("Player ", player.name, " has ", len(player.bankrollSnapshots), " bankroll snapshots")
            playerNames.append(player.name)
        ax.set_title("Plot of players' bankroll over time in a blackjack game of "+str(numHands)+" rounds")
        ax.set_xlabel("Round number")
        ax.set_ylabel("Bankroll ($)")
        
        # Add the legend
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), title="Players")
        plt.show()

class BlackJackGame:
    def __init__(self, shoeSize, bankroll, hands, tableMin, penetration, houseRules):
        print("Initializing game...")
        self.shoeSize = shoeSize
        self.numHands = hands
        self.tableMin = tableMin

        print("Dealer has rules: ")
        print("Deck Penetration %: ", penetration, " | Minimum table bet: $", tableMin)
        self.dealer = Dealer(penetration, shoeSize, houseRules, CasinoStrategy(houseRules, isCounting=False, accuracy=1))

        self.players = [Player("Counting with 1-6 Bet Spread", bankroll, BasicStrategy(houseRules, isCounting=True, accuracy=1), spread1_6()),
                        Player("Counting with 1-50 Bet Spread", bankroll, BasicStrategy(houseRules, isCounting=True, accuracy=1), spread1_50()),
                        Player("Perfect Basic Strategy", bankroll, BasicStrategy(houseRules, isCounting=False, accuracy=1), spread1_6()),
                        Player('99% Accurate Basic Strategy', bankroll, BasicStrategy(houseRules, isCounting=False, accuracy=0.99), spread1_50()),
                        Player('95% Accurate Basic Strategy', bankroll, BasicStrategy(houseRules, isCounting=False, accuracy=0.95), spread1_50()),
                        Player('75% Accurate Basic Strategy', bankroll, BasicStrategy(houseRules, isCounting=False, accuracy=0.75), spread1_50()),
                        Player('Casino Rules', bankroll, CasinoStrategy(houseRules, isCounting=False, accuracy=1), spread1_6()),
                        Player("Random", bankroll, RandomStrategy(houseRules, isCounting=False, accuracy=1), spread1_6())]
        print("There are ", len(self.players), " players in the game.")
    
    def clearAllCards(self, players: List[Player]):
        # Collect the cards from each player before moving onto the next round and put the cards in the
        # discard pile
        for player in players:
            print("clearAllCards() -> ", player.name, " has ", len(player.hands), " hands before clear.")
            print("Player hands: ", player.hands)
            for hand in player.hands:
                print("Discarding hand:", hand)
                hand.printHand(player.name)
                self.dealer.discardPlayersCards(hand, player.name)
            
            player.clearAllHands()
            print("clearAllCards() -> ", player.name, " has ", len(player.hands), " hands after clear.")
        
        # Discard the dealer's cards and move them to the discard pile
        self.dealer.discardDealersCards()

    def dealDealersHand(self, count):
        # Deal out the dealers cards
        upcard = self.dealer.dealCard()
        self.dealer.setUpCard(upcard)
        count.updateRunningCount(upcard.getValue())
        # The hidden card is not added to the count yet as only the dealer knows this information
        hiddenCard = self.dealer.dealCard()
        dealerHand = Hand([upcard, hiddenCard], 0)
        self.dealer.updateHand(dealerHand)
        print("Dealer shows:")
        upcard.printCard()
        print("Dealer hides:")
        hiddenCard.printCard()

    def dealPlayersHands(self, players, count):
        print("Dealing hands...")
        for player in players:
            betSize = player.calculateBetSize(self.tableMin, self.getTrueCount(count))

            card1: Card = self.dealer.dealCard()
            card2: Card = self.dealer.dealCard()

            count.updateRunningCount(card1.getValue())
            count.updateRunningCount(card2.getValue())

            player.updateBankroll(-1 * betSize)
            player.updateHand(Hand([card1, card2], betSize))
            player.getStartingHand().printHand(player.name)
    
    def doubleDown(self, player: Player, hand: Hand, count: HiLoCount):
        print("Doubling down!")
        player.updateBankroll(-1 * hand.getInitialBet())
        hand.doubleDown()
        self.hit(player, hand, count)
    
    def getTrueCount(self, count: HiLoCount):
        decksRemaining = self.dealer.shoe.getDecksRemaining()
        trueCount = count.getTrueCount(decksRemaining)
        return trueCount
    
    def handleBustHand(self, player: Player, hand: Hand):
        print("Hand went bust.")
        self.dealer.updateGains(hand.getInitialBet())
        self.dealer.discardPlayersCards(hand, player.name)
        player.clearHand(hand)
        

    def handleDealerBlackjack(self, players: List[Player], count: HiLoCount):
        # Need to update the count as the dealer reveals the hidden card to show blackjack
        # Guaranteed to have a count value of -1
        count.updateRunningCount(10)

        for player in players:
            for hand in player.hands:
                if hand.isBlackjack():
                    print("Player ", player.name, " pushes with another blackjack.")
                    player.updateBankroll(hand.betSize)
                elif hand.isInsured:
                    print("Player ", player.name, "'s hand is insured!")
                    player.updateBankroll(hand.betSize + hand.insuranceBet)
                else:
                    self.dealer.updateGains(hand.betSize)
    
    def handleInsurance(self, players: List[Player], count: HiLoCount):
        print("Dealer shows ace - Insurance offered")
        trueCount = self.getTrueCount(count)
        for player in players:
            for hand in player.hands:
                if player.strategy.willTakeInsurance(trueCount) and not hand.isBlackjack():
                    player.updateBankroll(-1 * hand.betSize / 2)
                    hand.insureHand()
                    print("Player ", player.name, " has insured their hand.")
        print("Insurance closed.")

    def handlePlayerBlackjack(self, player: Player, hand: Hand):
        payout = self.dealer.handlePayout(hand.betSize, isBlackjack=True)
        print("Blackjack! Initial bet: $", hand.getInitialBet(), " Payout: $", payout)
        player.updateBankroll(hand.betSize + payout)
        self.dealer.discardPlayersCards(hand, player.name)
        player.clearHand(hand)
    
    def handleRemainingHands(self, players: List[Player]):
        dealerValue = self.dealer.hand.finalHandValue
        for player in players:
            for hand in player.hands:
                print(player.name, " has ", hand.finalHandValue, " against the dealer's ", dealerValue)
                if hand.finalHandValue > dealerValue:
                    print("Player wins!")
                    payout = self.dealer.handlePayout(hand.betSize, isBlackjack=False)
                    print("Initial bet: $", hand.getInitialBet(), " Payout: $", payout)
                    player.updateBankroll(hand.betSize + payout)
                elif hand.finalHandValue < dealerValue:
                    print("Player loses!")
                    self.dealer.updateGains(hand.betSize)
                else:
                    print("Player pushes.")
                    player.updateBankroll(hand.betSize)

    def handleSplitPair(self, player: Player, hand: Hand, dealerUpcard: Card, count: HiLoCount):
        print("Determining whether or not to split pair based on player's strategy...")
        trueCount = self.getTrueCount(count)
        if player.strategy.shouldSplitPair(hand.getHandValue() / 2, dealerUpcard.getValue()) and player.calculateBetSize(self.tableMin, trueCount) <= player.bankroll:
            print("Splitting pair!")
            splitHand = player.splitPair(hand)
            return splitHand
        print("Player decided not to split pair.")
        return None

    
    def hit(self, player: Player, hand: Hand, count: HiLoCount):
        hitCard = self.dealer.dealCard()
        count.updateRunningCount(hitCard.getValue())
        hand.addCard(hitCard)
        print(player.name, " has new hand: ")
        hand.printHand(player.name)

    def playDealerHand(self, count):
        print("Dealer is now playing their hand...")
        action: GameActions = None
        softTotalDeductionCount = 0

        while (action != GameActions.STAND.value):
            if self.dealer.hand.isBust():
                if softTotalDeductionCount < self.dealer.hand.getAcesCount():
                    print("Dealer busts! Ace now becomes 1. Old hand value: ", self.dealer.hand.getHandValue(), " New value: ", self.dealer.hand.getHandValue() - 10)
                    softTotalDeductionCount += 1
                else:
                    print("Dealer busts! All players are rewarded")
                    break
            if self.dealer.hand.isSoftTotal(softTotalDeductionCount) and softTotalDeductionCount < self.dealer.hand.getAcesCount():
                print("Dealer has soft total.")
                action = self.dealer.strategy.softTotalOptimalDecision(self.dealer.hand, self.dealer.upcard, softTotalDeductionCount)
            else:
                print("Dealer has hard total...")
                action = self.dealer.strategy.hardTotalOptimalDecision(self.dealer.hand, self.dealer.upcard.getValue(), softTotalDeductionCount)
            
            if (action == GameActions.HIT.value):
                print("Dealer hits...")
                hitCard = self.dealer.dealCard()
                count.updateRunningCount(hitCard.getValue())
                self.dealer.hand.addCard(hitCard)
                print("Dealer now has hand: ")
                self.dealer.hand.printHand("Dealer")
            elif (action == GameActions.STAND.value):
                print("Dealer will stand...")
                self.dealer.hand.setFinalHandValue(self.dealer.hand.getHandValue() - softTotalDeductionCount * 10)
                break

            

    def playHands(self, player: Player, dealerUpcard: Card, handNumber, count):
        dealtHand = player.getStartingHand()
        print(player.name, " is playing their hand...")

        # Check if the dealt hand is a blackjack and payout immediately if it is
        if dealtHand.isBlackjack():
            self.handlePlayerBlackjack(player, dealtHand)
        else:
            # First, determine if we have a pair and if we should split:
            # if dealtHand.isPair():
            #     self.handleSplitPair(player, dealtHand, dealerUpcard, count)
            
            # It's now possible that we have two hands that we need to simulate as the player could have split the pair.
            # Instatiate a new dealtHands object:
            # handQueue = deque()
            # for hand in player.hands:
            #     handQueue.append(hand)

            handQueue = deque()
            handQueue.append(player.getStartingHand())

            while len(handQueue) > 0:
                hand = handQueue.pop()
                action: GameActions = None
                softTotalDeductionCount = 0

                while (action != GameActions.STAND.value):
                    if hand.isBust():
                        if softTotalDeductionCount < hand.getAcesCount():
                            print("BUST! Ace now becomes 1. Old hand value: ", hand.getHandValue(), " New value: ", hand.getHandValue() - 10)
                            softTotalDeductionCount += 1
                        else:
                            print("BUST! Value is: ", hand.getHandValue() - softTotalDeductionCount * 10)
                            self.handleBustHand(player, hand)
                            break
                    if hand.isPair():
                        print("We have a pair...")
                        splitHand = self.handleSplitPair(player, hand, dealerUpcard, count)
                        if splitHand is not None:
                            handQueue.append(splitHand)
                            handQueue.append(hand)
                            break
                    
                    if hand.isSoftTotal(softTotalDeductionCount) and softTotalDeductionCount < hand.getAcesCount():
                        print("We have a soft total...")
                        action = player.strategy.softTotalOptimalDecision(hand, dealerUpcard.getValue(), softTotalDeductionCount)
                    else:
                        # Get hard total value
                        print("We have a hard total of ", hand.getHandValue()- softTotalDeductionCount * 10)
                        action = player.strategy.hardTotalOptimalDecision(hand, dealerUpcard.getValue(), softTotalDeductionCount)
                    if (action == GameActions.HIT.value):
                        print("Player is gonna hit!")
                        self.hit(player, hand, count)
                    elif (action == GameActions.STAND.value):
                        print("Player will stand")
                        hand.setFinalHandValue(hand.getHandValue() - softTotalDeductionCount * 10)
                        break
                    elif (action == GameActions.DOUBLE.value):
                        print("Double down!")
                        self.doubleDown(player, hand, count)
                    
        print(player.name, " has played all of their hands!")
    
    def printRoundInformation(self, players: List[Player], count: HiLoCount, roundNumber: int):
        print(" - - - - - - - - - - -")
        print(" - - - - - - - - - - -")
        print("Round: ", roundNumber, " Running Count: ", count.runningCount)
        print(" - - - - - - - - - - -")
        print(" - - - - - - - - - - -")
        for player in players:
            prevIndex = len(player.bankrollSnapshots) - 2
            print(player.name, ' has a bankroll of $', player.bankroll, " (Prev hand: $", player.bankrollSnapshots[prevIndex], ")")
    
    def startGame(self, isVerbose):
        self.dealer.shuffle()
        print("Starting new blackjack game:")

        handCount = 1
        playersInGame: List[Player] = []
        playersInBreak: List[Player] = []

        for player in self.players:
            playersInGame.append(player)
            print("Player: ", player.name, " has joined the game.")

        count = HiLoCount()

        # Play the game! 
        while (handCount <= self.numHands and len(playersInGame) > 0):
            if isVerbose:
                self.printRoundInformation(playersInGame, count, handCount)
            # Deal out the players' and dealer's cards
            self.dealPlayersHands(playersInGame, count)
            self.dealDealersHand(count)

            # If the dealer shows an ace, dealer will offer insurance to all players.
            if self.dealer.insuranceIsOffered():
                self.handleInsurance(playersInGame, count)
            
            # If the dealer was dealt a blackjack, all players automatically lose UNLESS they too have a blackjack
            if self.dealer.hand.isBlackjack():
                self.handleDealerBlackjack(playersInGame, count)
            else:
                # Allow players to play out each of their hands
                for player in playersInGame:
                    self.playHands(player, self.dealer.upcard, handCount, count)
            
            # Now, the dealer will play out their hand
            self.playDealerHand(count)

            # Next, determine which existing hands beat the dealer and perform all necessary payouts
            self.handleRemainingHands(playersInGame)
        
            handCount = handCount + 1

            self.clearAllCards(playersInGame)

            # Used to debug deck sizes to ensure that no cards are being lost:
            self.dealer.ensureDeckCompleteness(isVerbose=True)
            
            # If we have exceeded or reached optimal shoe penetration, reset the shoe and the running count
            if self.dealer.deckPenetrationTooHigh():
                self.dealer.shuffle()
                count.resetCount()
            
            
            for player in playersInGame:
                player.takeBankrollSnapshot()
                if player.bankroll < self.tableMin:
                    print(player.name, " has gone broke and is out of the game.")
                    playersInGame.remove(player)




if __name__ == '__main__':
    main()