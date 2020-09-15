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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Get all the legal Actions and store it so that we can use for next directions.
        legalDirections = gameState.getLegalActions()

        # From all the directions we need to choose which will best suite the pacman
        scoreOfPacman = [self.evaluationFunction(gameState, action) for action in legalDirections]
        # Here we are storing the maximum score we get because the maximum value will help us reach the gola state quickly
        maxScoreOfPacman = max(scoreOfPacman) 
        # From all the movements select the one which will help us reach the goal state
        indexMovement = [movement for movement in range(len(scoreOfPacman)) if scoreOfPacman[movement] == maxScoreOfPacman]
        chosenIndex = random.choice(indexMovement) # Pick randomly among the best

        "Add more of your code here if you want to"
        # return the direction which has the index that needs to be choosen
        return legalDirections[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        """ We are using manhattanDistance that has been omported from util library to get the distacnes of the ghost from the 
            path and storing it in a variable.
            Also this helps in knowing the distance from the distance
        """
        ghostDistances = [manhattanDistance(newPos, state.getPosition()) for state in newGhostStates]
        """
        Here we are finding the closest distance of ghost by taking into account the minimum distance from the array of distances
        """
        minGhostDistance = min(ghostDistances)

        
        """
        The gotScore variable will have periodic scores to check if the score has increased or not
        """
        gotScore = successorGameState.getScore() - currentGameState.getScore()

        """
        Here we are getting all the food distances and food loaction in the path as well as finding the closest updating 
        the distance to find the shortest path
        """
        position = currentGameState.getPacmanPosition()
        """
        Finding the food loaction in the path
        """
        foodLocation = currentGameState.getFood().asList()
        """
        Getting all the food distance array and getting th eminimum distance
        """
        foodDistances = [manhattanDistance(position, food) for food in foodLocation]
        closestFoodDistance = min(foodDistances)

        """
        Here we are getting the minimum food distance and update the nearest food distance in the path
        """
        updatedFoodsDistances = [manhattanDistance(newPos, food) for food in foodLocation]
        updatedNearestFoodDistance = 0 if not updatedFoodsDistances else min(updatedFoodsDistances)

        isNearer = closestFoodDistance - updatedNearestFoodDistance

        """
        After calculating the values of ghost and food it is necessary to update the direction of pacman and hence we are getting the direction of the pacman
        """

        direction = currentGameState.getPacmanState().getDirection()

        """
        This the reflex formula which  we created as per the algorithm
        """

        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        """
        The score we got needs to be greater than 0
        Need to check if the ghost are near or no
        """
        if gotScore > 0:
            return 8
        elif isNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
      Your minimax agent (question 2)
    """
    """
    Minimax is a decision rule used in artificial intelligence, decision theory, game theory, statistics and philosophy for minimizing the possible loss for a worst case (maximum loss) scenario
    """
    def minMax(self, gameState,index):
        length_list = []
        """
        This is the terminal state to check if the game has been won, lost or what is the decision.
        """
        if index == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return ('',self.evaluationFunction(gameState))
        
        else:
            # We generate the newGameStates for all legal actions and calculate the respective MinMax value
            """
            Generating all the legal actions so that the pacmna can estimate which actions to choose
            """
            estimate = index%gameState.getNumAgents()
            for action in gameState.getLegalActions(estimate):
                newGameState = gameState.generateSuccessor(estimate, action)
                """
                Going to the next index as we are going one level down
                """
                direction,values = self.minMax(newGameState, index+1)          
                """
                Appending the legal moves and the value calculated 
                """
                length_list.append((action,values))    
            """
            #Max state will be when the modulo of index to number of Agents is 0
            """                                  
            if(estimate ==0):          
                """
                Assigning the most minimum value to the max variable
                """
                maxValue = -1000000                
                maxAction = 0
                """
                Calculating the maximum value and action and returning it to the next stage
                """
                for action,value in length_list:                                      
                    if value>maxValue:
                        maxValue = value
                        maxAction = action
                return (maxAction,maxValue)
            else:
                """
                Assigning the most maximum value to the min variable
                When the Ghost moves the remaining states will be Min states
                """
                minValue = 1000000                                            
                minAction = 0
                """
                Calculating the Minimum value and action and returning it to the next stage
                """
                for action,value in length_list:                                      
                    if value<minValue:
                        minValue = value
                        minAction = action
                return (minAction,minValue)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        direction,value = self.minMax(gameState, self.index)
        """
        Return the direction in which the pacman should move
        """
        return direction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def maxValue(self, gameState, alpha, beta, index):
        """
        Assigning the most minimum value for calculation
        """
        values = -1000000
        maxAction = 0
        estimate = index%gameState.getNumAgents()
        for action in gameState.getLegalActions(estimate):
            """
            Generating the successor from the current position to advance the path
            """
            newGameState = gameState.generateSuccessor(estimate, action)
            direction,value = self.getVal(newGameState, alpha, beta, index+1)
            """
            Updating the values to increase the efficiency of pacman
            """
            if(value>=values):
                values = value
                maxAction = action
            """
            If current value is greater than beta then we prune the remaining branches and return the value
            """
            if values > beta:                                   
                return (action,values)
            """
             Updating the alpha value is necessary to score the max value
            """
            alpha = max(values,alpha)                         
        return (maxAction,values)
    
    def minValue(self, gameState, alpha, beta, index):
        """
        Assigning the most maximum value for calculation
        """
        values = 1000000
        minAction = 0
        estimate = index%gameState.getNumAgents()
        for action in gameState.getLegalActions(estimate):
            """
            Generating the successor from the current position to advance the path
            """
            newGameState = gameState.generateSuccessor(estimate, action)
            direction,value = self.getVal(newGameState, alpha, beta, index+1)
            """
            Updating the action and values of pacman
            """
            if(value<=values):
                values = value
                minAction = action
            """
            # if current value is less than alpha we prune the remaining branches and return the value
            """
            if values < alpha:                                 
                return (action,values) 
            """
             Updating the beta value is necessary to score the min value
            """
            beta = min(values,beta)                                                    
        return (minAction,values)
    
    def getVal(self, gameState, alpha, beta, index):
        """
        Check wheather the goal state is reached has the pacmna won, lost or what is the state of the agent
        """
        if index == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return ('',self.evaluationFunction(gameState))
        if(index%gameState.getNumAgents()==0):
            return self.maxValue(gameState, alpha, beta, index)
        else:
            return self.minValue(gameState, alpha, beta, index)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        direction,value = self.getVal(gameState, -100000, 1000000, self.index)
        return direction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        """
        Implementing the Expectimax algorithm as per the logic.
        """

        def maxValue(state, depth):
            """ Here we are getting all the legal actions"""
            legalActions = state.getLegalActions(0)
            """ Return the state in which the function is implemented"""
            if not legalActions or depth == self.depth:
                return self.evaluationFunction(state)
            """
            The expectimax algorithm is used to increase the scores
            """
            v = max(expValue(state.generateSuccessor(0, action), 0 + 1, depth + 1) for action in legalActions)
            return v

        def expValue(state, agentIndex, depth):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            """
            Here the probability we are calculating since while finding the maximum for new state or depth it wll help
            in predicting in which state to choose
            """
            probability = 1.0 / len(legalActions)
            v = 0
            for action in legalActions:
                """
                Getting all the successors from the current path
                """
                newState = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    """
                    Updating the values for increasing the efficiency of pacman
                    """
                    v += maxValue(newState, depth) * probability
                else:
                    v += expValue(newState, agentIndex + 1, depth) * probability
            return v

        legalActions = gameState.getLegalActions()
        """
        Lambdas are one line functions. They are also known as anonymous functions in some other languages.
        """
        bestMovement=max(legalActions, key=lambda action: expValue(gameState.generateSuccessor(0, action), 1, 1))
        return bestMovement


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    position = currentGameState.getPacmanPosition()
    foodPosition = currentGameState.getFood().asList()
    """
    Getting all the position of foods
    """
    closestFoodDis = min(manhattanDistance(position, food) for food in foodPosition) if foodPosition else 0.5
    """
    Calculating the score of pacman game
    """
    score = currentGameState.getScore()

    """
      Sometimes pacman will stay there and there only even when there's a food besides it, because 
      stop action has the same priority with other actions, that needs to be improved
    """
    evaluation = 1.0 / closestFoodDis + score
    return evaluation


# Abbreviation
better = betterEvaluationFunction
