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
import heapq

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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    fringe=util.Stack()
    visited=[]
    #visited里存的是元组表示坐标
    fringe.push((problem.getStartState(),[]))
    #栈里的元素是一个元组((x,y),[action]) x,y为该点坐标，[action]存储到这个点的路径
    
    while fringe.isEmpty!=True:
        #弹出的时候才标记访问过
        #!!与bfs不同。dfs在元素弹出的时候才默认访问（走到）；而bfs是加入的时候就已经被扩展了
        state,action=fringe.pop()
        visited.append(state)
        if problem.isGoalState(state):
            return action
        for next in problem.getSuccessors(state):
            n_state=next[0]
            n_action=next[1]
            if n_state not in visited:
                fringe.push((n_state,action+[next[1]]))
    


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # 构建⼀个队列
    Frontier = util.Queue()
    # 创建已访问节点集合
    Visited = []
    # 将(初始节点,空动作序列)⼊队
    Frontier.push( (problem.getStartState(), []) )
    # 将初始节点标记为已访问节点
    Visited.append( problem.getStartState() )
    # 判断队列⾮空
    while Frontier.isEmpty() == 0:
        state, actions = Frontier.pop()
        # 判断是否为⽬标状态，若是则返回到达该状态的累计动作序列
        if problem.isGoalState(state):
            return actions
        # 遍历所有后继状态
        for next in problem.getSuccessors(state):
        # 新的后继状态
            n_state = next[0]
        # 新的action
            n_direction = next[1]
        # 若该状态没有访问过
            if n_state not in Visited:
        # 计算到该状态的动作序列，⼊队
                Frontier.push( (n_state, actions + [n_direction]) )
                Visited.append( n_state )

    #util.raiseNotDefined()
    
#! 例题答案如下
# def breadthFirstSearch(problem):
#     """Search the shallowest nodes in the search tree first."""
#     #python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs --frameTime 0
#     #python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5 --frameTime 0
#     "*** YOUR CODE HERE ***"

#     Frontier = util.Queue()
#     Visited = []
#     Frontier.push( (problem.getStartState(), []) )
#     #print 'Start',problem.getStartState()
#     Visited.append( problem.getStartState() )

#     while Frontier.isEmpty() == 0:
#         state, actions = Frontier.pop()
#         if problem.isGoalState(state):
#             #print 'Find Goal'
#             return actions 
#         for next in problem.getSuccessors(state):
#             n_state = next[0]
#             n_direction = next[1]
#             if n_state not in Visited:
                
#                 Frontier.push( (n_state, actions + [n_direction]) )
#                 Visited.append( n_state )

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe=util.PriorityQueue()#优先队列，存的是((x,y),[action],cost)坐标，到坐标的路线，到该点的cost
    visited=[]#记录已经访问过的节点，存的是元组(x,y)
    fringe.push((problem.getStartState(),[],0),0)
    while fringe.isEmpty()!=True:
        nowState,nowAction,nowCost=fringe.pop()
        if problem.isGoalState(nowState):#找到了
            print(nowAction)
            return nowAction
        
        #如果不判断这一步，可能出现已经出了队列的节点（被访问过的）再一次执行下面这些操作的情况，而下面这些操作调用了
        #getSuccessors，尽管实际上不会有任何n_state被加入队列，但是在autograder看来是多余的操作。
        if nowState in visited:
            continue
        
        visited.append(nowState)#弹出的时候才访问！
        for next in problem.getSuccessors(nowState):
            n_state=next[0]#下一个点的坐标
            n_direction=next[1]#前往下一个点的方向
            if n_state not in visited:#这个节点可以被加入优先队列
                n_cost=next[2]+nowCost#到达该节点的费用
                fringe.update((n_state,nowAction+[n_direction],n_cost),n_cost)



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe=util.PriorityQueue()
    #优先队列，存的是((x,y),[action],cost)坐标，到坐标的路线，到该点的cost,priority是cost+heuristic
    visited=[]#记录已经访问过的节点，存的是元组(x,y)
    startHeuristic=heuristic(problem.getStartState(),problem)
    fringe.push((problem.getStartState(),[],0),startHeuristic)
    while fringe.isEmpty()!=True:
        nowState,nowAction,nowCost=fringe.pop()
        if problem.isGoalState(nowState):#找到了
            return nowAction
        
        #???visited标记的是已经访问过的，优先队列中会存在两个相同节点优先级不同吗
        if nowState in visited:
            continue

        visited.append(nowState)#弹出的时候才访问！
        for next in problem.getSuccessors(nowState):
            n_state=next[0]#下一个点的坐标
            n_direction=next[1]#前往下一个点的方向
            n_cost=next[2]#前往下一个点的花费
            if n_state not in visited:#这个节点可以被加入优先队列
                n_priority=n_cost+nowCost+heuristic(n_state,problem)
                fringe.update((n_state,nowAction+[n_direction],n_cost+nowCost),n_priority)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
