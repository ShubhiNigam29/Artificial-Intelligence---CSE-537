# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

cost = 0
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def graphSolution(problem,s_node, visited_node,stack,final_path,overall_cost):
    """
    Here we have implemented a common function for bfs and dfs wherein we pass the parameters
    and implement the function.
    """
    while (problem.isGoalState(s_node)!= True):             # Making sure tto check is the goal state reached
         successor_nodes = problem.getSuccessors(s_node)    # getting all the successor nodes from the present position
         length_list = len(final_path)                      # getting the total length of the list  so that we find the direction node
         for l,m,n in successor_nodes:
             temp = final_path[0:length_list]
             temp.append(m)
             stack.push((l,temp))                           # pusing the node and the direction in which the pacman can move
         while (problem.isGoalState(s_node)!= True):        # traversing the other nodes in the list and check if we have reached the goal state
             s_node,final_path = stack.pop()                # so that traversing can be minimized
             if(s_node not in visited_node):
                 visited_node.append(s_node)                 
                 break
    return final_path
    

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"""
    
    cost = 0
        
    return graphSolution(problem,problem.getStartState(),[],util.Stack(),[],cost)
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first.
    "*** YOUR CODE HERE ***"
    """
    queue = util.Queue()     # Creeating a queue to get the first nodes inside the queue
    s_node = problem.getStartState()   #
    queue.push((s_node, []))    # Pushing the first node into the queue
    final_path = set()          # Storing all the nodes for the path from the start state to the goal state
    final_path.add(s_node)

    while not queue.isEmpty():
        present_node = queue.pop()      # remove the first node from the queue and explore the other successor nodes
        if problem.isGoalState(present_node[0]):    # check if the first node is the goal state or not
            return present_node[1]
        successors = problem.getSuccessors(present_node[0])
        for nodes in successors:
            if nodes[0] in final_path:
                continue
            final_path.add(nodes[0])
            queue.push((nodes[0], present_node[1] + [nodes[1]]))    # Push the node if the updates weights

    return None
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    cost = 0
    s_node = problem.getStartState()
    stack = util.PriorityQueue()
    visited_node = []
    final_path = []
    while (problem.isGoalState(s_node)!= True):     # Making sure tto check is the goal state reached
        successor_nodes = problem.getSuccessors(s_node) # getting all the successor nodes from the present position
        length_list = len(final_path)           # getting the total length of the list  so that we find the direction node
        for l,m,n in successor_nodes:
            temp = final_path[0:length_list]
            temp.append(m)
            temp_cost = cost + n
            stack.push((l,temp,temp_cost),temp_cost)  # pusing the node and the direction in which the pacman can move
        while (problem.isGoalState(s_node)!= True):  # traversing the other nodes in the list and check if we have reached the goal state
            s_node,final_path,cost = stack.pop()    # so that traversing can be minimized
            if(s_node not in visited_node):
                visited_node.append(s_node)                 
                break
    return final_path
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    s_node = problem.getStartState()          
    visited_node = [s_node]                      
    cost = 0
    return graphSolutionForUcsAstar(problem,problem.getStartState(),visited_node,util.PriorityQueue(),[],cost,heuristic)

def graphSolutionForUcsAstar(problem,s_node,visited_node,queue,final_path,cost,heuristic):
    '''
    Here it is similar to UCS but here we have given with heuristic value
    So we are adding the heuristic cost to the path cost and adding those nodes in a priority queue
    '''
    while( problem.isGoalState(s_node) != True ):
        successor_nodes = problem.getSuccessors(s_node)
        length_list = len(final_path)
        for node,direction,n_cost in successor_nodes:
            temp = final_path[0:length_list]
            temp.append(direction)
            temp_cost = cost + n_cost                   #Calculating the path cost with the current cost and the node cost
            heuristic_cost = heuristic(node,problem)   #Calculating the heuristic cost and adding it to the path cost
            path_cost = cost + n_cost
            temp_cost = path_cost + heuristic_cost
            queue.push((node,temp,path_cost),temp_cost)
        while (problem.isGoalState(s_node)!= True):
            s_node,final_path,cost = queue.pop()      #Getting the successor to be explored ,direction from start state and cost to the path
            if s_node not in visited_node:            #If the successor is in the visited set get the next successor
                visited_node.append(s_node)
                break
    return final_path
    

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
