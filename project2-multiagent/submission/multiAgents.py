# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        This question is not included in project for CSCI360
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return childGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumGhost():
        Returns the total number of ghosts in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # retuns list [value, direction]
        return self.value(1, gameState, 0)[1]
        util.raiseNotDefined()

    def value(self, depth, gameState, currentAgent):
        # terminal state return state's utility
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # agent is max (pacman)
        if currentAgent == 0:
            return self.maxValue(depth, gameState, currentAgent)
        # agent is min (ghost)
        else:
            return self.minValue(depth, gameState, currentAgent)

    def maxValue(self, depth, gameState, currentAgent):
        # need to store num for value and direction
        v = [float('-inf'), None]

        for action in gameState.getLegalActions(currentAgent):
            successor = gameState.getNextState(currentAgent, action)
            itV = self.value(depth, successor, currentAgent + 1)

            # itV is list when it's returned from max/minValue
            # it's float if it's returned from reaching terminal state
            if isinstance(itV, list):
                if itV[0] > v[0]:
                    v[0] = itV[0]
                    v[1] = action
            else:
                if itV > v[0]:
                    v[0] = itV
                    v[1] = action
        return v

    def minValue(self, depth, gameState, currentAgent):
        # need to store num for value and direction
        v = [float('inf'), None]

        # Check whether to call maxValue (pacman) or call minValue (ghost)
        # for next agent
        nextAgent = currentAgent + 1
        if nextAgent > gameState.getNumGhost():
            nextAgent = 0

        for action in gameState.getLegalActions(currentAgent):
            successor = gameState.getNextState(currentAgent, action)
            # nextAgent is back to pacman
            if nextAgent == 0:
                # reached the end, evaluate gamestate
                if depth == self.depth:
                    itV = self.evaluationFunction(successor)
                # depth + 1 since going to pacman (min -> max)
                else:
                    itV = self.value(depth + 1, successor, nextAgent)
            else:
                # depth doesn't change since looking at next ghost (min -> min)
                itV = self.value(depth, successor, nextAgent)

            # itV is list when it's returned from max/minValue
            # it's float if it's returned from reaching terminal state
            if isinstance(itV, list):
                if itV[0] < v[0]:
                    v[0] = itV[0]
                    v[1] = action
            else:
                if itV < v[0]:
                    v[0] = itV
                    v[1] = action

        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # retuns list [value, direction]
        return self.value(1, gameState, 0, float('-inf'), float('inf'))[1]
        util.raiseNotDefined()

    def value(self, depth, gameState, currentAgent, alpha, beta):
        # terminal state return state's utility
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # agent is max (pacman)
        if currentAgent == 0:
            return self.maxValue(depth, gameState, currentAgent, alpha, beta)
        # agent is min (ghost)
        else:
            return self.minValue(depth, gameState, currentAgent, alpha, beta)

    def maxValue(self, depth, gameState, currentAgent, alpha, beta):
        # need to store num for value and direction
        v = [float('-inf'), None]

        for action in gameState.getLegalActions(currentAgent):
            successor = gameState.getNextState(currentAgent, action)
            itV = self.value(depth, successor, currentAgent + 1, alpha, beta)

            # itV is list when it's returned from max/minValue
            # it's float if it's returned from reaching terminal state
            if isinstance(itV, list):
                if itV[0] > v[0]:
                    v[0] = itV[0]
                    v[1] = action
            else:
                if itV > v[0]:
                    v[0] = itV
                    v[1] = action

            if v[0] > beta:
                return v
            alpha = max(alpha, v[0])
        return v

    def minValue(self, depth, gameState, currentAgent, alpha, beta):
        # need to store num for value and direction
        v = [float('inf'), None]

        # Check whether to call maxValue (pacman) or call minValue (ghost)
        # for next agent
        nextAgent = currentAgent + 1
        if nextAgent > gameState.getNumGhost():
            nextAgent = 0

        for action in gameState.getLegalActions(currentAgent):
            successor = gameState.getNextState(currentAgent, action)
            # nextAgent is back to pacman
            if nextAgent == 0:
                # reached the end, evaluate gamestate
                if depth == self.depth:
                    itV = self.evaluationFunction(successor)
                # depth + 1 since going to pacman (min -> max)
                else:
                    itV = self.value(depth + 1, successor, nextAgent, alpha, beta)
            else:
                # depth doesn't change since looking at next ghost (min -> min)
                itV = self.value(depth, successor, nextAgent, alpha, beta)

            # itV is list when it's returned from max/minValue
            # it's float if it's returned from reaching terminal state
            if isinstance(itV, list):
                if itV[0] < v[0]:
                    v[0] = itV[0]
                    v[1] = action
            else:
                if itV < v[0]:
                    v[0] = itV
                    v[1] = action

            if v[0] < alpha:
                return v

            beta = min(beta, v[0])

        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # retuns list [value, direction]
        return self.value(1, gameState, 0)[1]
        util.raiseNotDefined()

    def value(self, depth, gameState, currentAgent):
        # terminal state return state's utility
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # agent is max (pacman)
        if currentAgent == 0:
            return self.maxValue(depth, gameState, currentAgent)
        # agent is min (ghost)
        else:
            return self.minValue(depth, gameState, currentAgent)

    def maxValue(self, depth, gameState, currentAgent):
        # need to store num for value and direction
        v = [float('-inf'), None]

        for action in gameState.getLegalActions(currentAgent):
            successor = gameState.getNextState(currentAgent, action)
            itV = self.value(depth, successor, currentAgent + 1)

            # itV is list when it's returned from max/minValue
            # it's float if it's returned from reaching terminal state
            if isinstance(itV, list):
                if itV[0] > v[0]:
                    v[0] = itV[0]
                    v[1] = action
            else:
                if itV > v[0]:
                    v[0] = itV
                    v[1] = action
        return v

    def minValue(self, depth, gameState, currentAgent):
        # need to store num for value and direction
        v = [0, None]

        # Check whether to call maxValue (pacman) or call minValue (ghost)
        # for next agent
        nextAgent = currentAgent + 1
        if nextAgent > gameState.getNumGhost():
            nextAgent = 0

        totalActions = len(gameState.getLegalActions(currentAgent))

        for action in gameState.getLegalActions(currentAgent):
            successor = gameState.getNextState(currentAgent, action)
            p = 1 /totalActions
            # nextAgent is back to pacman
            if nextAgent == 0:
                # reached the end, evaluate gamestate
                if depth == self.depth:
                    itV = self.evaluationFunction(successor)
                # depth + 1 since going to pacman (min -> max)
                else:
                    itV = self.value(depth + 1, successor, nextAgent)
            else:
                # depth doesn't change since looking at next ghost (min -> min)
                itV = self.value(depth, successor, nextAgent)

            # itV is list when it's returned from max/minValue
            # it's float if it's returned from reaching terminal state
            if isinstance(itV, list):
                v[0] += p * itV[0]
            else:
                v[0] += p * itV

        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 4).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    numFood = currentGameState.getNumFood()

    foodDist = 0
    # reward for taking both food
    if numFood == 0:
        score += 1000
    else:
        # get the dist of all remaining food
        for food in newFood.asList():
            foodDist += util.manhattanDistance(food, newPos)

    # get the dist of all ghosts
    ghostDist = 0
    for ghost in newGhostStates:
        # only care about the ones that are not scared since they gonna get munched on
        if newScaredTimes[newGhostStates.index(ghost)] == 0:
            ghostDist += util.manhattanDistance(ghost.getPosition(), newPos)

    # to avoid since ghost is super close by
    if ghostDist < 2:
        return -1000
    # to take since food is super close by
    if foodDist < 1:
        return 1000

    # score add inverse of ghostDist and foodDist for better comparison
    score += (1 / ghostDist) + (1 / foodDist)

    return score

    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
