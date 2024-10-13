"""
Author: Amin Babar
Date:   03/08/20
Player name: Anonymous


Implemented mulitple player classes for othello that use various combinations of IDS, 
minimax and alphabetapruning.

Player classes included: ShortTermMaximizer, ShortTermMinimizer, MinimaxPlayer, AlphaBetaPruning, 
IterativeDeepeningSearchPlayer, MontecarloTreeSearchPlayer, TimeRemainingIDSPlayer, BetterHeuristic, 
c_squares_heuristic, Superior, Move_ordering, ExperimentalPlayer1 and TournamentPlayer

The tournament player class is the same as BetterHeuristic class except for one small change.

Superior, Move_ordering and ExperimentalPlayer1 classes are still work in progress and are
possibly the next steps for me to make my game AI better. 

Since there is a lot of code that is repeated, only code that is new has been commented.

# CITE
Strategies from following websites:
1. http://radagast.se/othello/Help/strategy.html
2. http://www.soongsky.com/othello/en/strategy/parity.php
3. http://www.ffothello.org/livres/beginner-Randy-Fang.pdf

"""

from othello import *
import random, sys
from statistics import mean 
import time

class MoveNotAvailableError(Exception):
    """Raised when a move isn't available."""
    pass

class OthelloTimeOut(Exception):
    """Raised when a player times out."""
    pass


class OthelloPlayer():
    """Parent class for Othello players."""

    def __init__(self, color):
        assert color in ["black", "white"]
        self.color = color

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. Each type of player
        should implement this method. remaining_time is how much time this player
        has to finish the game."""
        pass



#################################################################################################################################
class RandomPlayer(OthelloPlayer):
    """Plays a random move."""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        return random.choice(state.available_moves())


#################################################################################################################################
class HumanPlayer(OthelloPlayer):
    """Allows a human to play the game"""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        available = state.available_moves()
        print("----- {}'s turn -----".format(state.current))
        print("Remaining time: {:0.2f}".format(remaining_time))
        print("Available moves are: ", available)
        move_string = input("Enter your move as 'r c': ")

        # Takes care of errant inputs and bad moves
        try:
            moveR, moveC = move_string.split(" ")
            move = OthelloMove(int(moveR), int(moveC), state.current)
            if move in available:
                return move
            else:
                raise MoveNotAvailableError # Indicates move isn't available

        except (ValueError, MoveNotAvailableError):
            print("({}) is not a legal move for {}. Try again\n".format(move_string, state.current))
            return self.make_move(state, remaining_time)




#################################################################################################################################


class ShortTermMaximizer(OthelloPlayer):
    """ Automatically choses a move for the player that has maximum score for current move"""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move with maximum score to make."""
        available = state.available_moves()

        score_change = []

        # Counts the number of pieces for the current player for each move and
        # appends them to a list
        for move in available:
            move = move.__repr__()
            moveR = move[1]
            moveC = move[3]

            new_state = copy.deepcopy(state)
            move = OthelloMove(int(moveR), int(moveC), state.current)
            new_state = new_state.apply_move(move)
            score_change.append(new_state.count(state.current))


        # The move with the max score for the current player is used
        index_max = score_change.index(max(score_change))
        move_max = (available[index_max]).__repr__()


        # Takes care of errant inputs and bad moves
        try:
            moveR = move_max[1]
            moveC = move_max[3]
            move = OthelloMove(int(moveR), int(moveC), state.current)
            if move in available:
                return move
            else:
                raise MoveNotAvailableError # Indicates move isn't available

        except (ValueError, MoveNotAvailableError):
            print("({}) is not a legal move for {}. Try again\n".format(move_max, state.current))
            return self.make_move(state, remaining_time)



#################################################################################################################################

class ShortTermMinimizer(OthelloPlayer):
    """ Automatically choses a move for the player that has minimum score for current move"""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move with minimum score to make."""
        available = state.available_moves()

        score_change = []

        # Counts the number of pieces for the current player for each move and
        # appends them to a list
        for move in available:
            move = move.__repr__()
            moveR = move[1]
            moveC = move[3]

            new_state = copy.deepcopy(state)
            move = OthelloMove(int(moveR), int(moveC), state.current)
            new_state = new_state.apply_move(move)
            score_change.append(new_state.count(state.current))


        # The move with the min score for the current player is used
        index_min = score_change.index(min(score_change))
        move_min = (available[index_min]).__repr__()


        # Takes care of errant inputs and bad moves
        try:
            moveR = move_min[1]
            moveC = move_min[3]
            move = OthelloMove(int(moveR), int(moveC), state.current)
            if move in available:
                return move
            else:
                raise MoveNotAvailableError # Indicates move isn't available

        except (ValueError, MoveNotAvailableError):
            print("({}) is not a legal move for {}. Try again\n".format(move_min, state.current))
            return self.make_move(state, remaining_time)




#################################################################################################################################

class MinimaxPlayer(OthelloPlayer):
    """
    Implements the minimax algorithm for depth 4. 
    """

    # Method that calls on find_move. Redundant.
    def make_move(self, state, remaining_time):
        return self.find_move(state)



    def find_move(self, state):
        """
        Calls the minimax algorithm on the current available moves. The depth used is 4.
        Returns move with max/min utility. 
        """

        depth = 4
        available = state.available_moves()

        # Black player is set as Max player
        if state.current == "black":
            is_Max = True
        else:
            is_Max = False

        new_state = copy.deepcopy(state)

        # Calls in the minimax_algorithm on each move in available. The utility returned
        # is appended to the moves_eval_list
        moves_eval_list = []
        for move in available:
            moves_eval_list.append(self.minimax_algorithm(new_state, 0, depth, move, is_Max))

        # Move with Maximum utility is returned if player max. Move with Minimum utility 
        # returned if player is minimum
        if is_Max:
            return available[moves_eval_list.index(max(moves_eval_list))]
        else:
            return available[moves_eval_list.index(min(moves_eval_list))]


    def minimax_algorithm(self, state, currDepth, depth, move, is_Max):
        """
        Implements the recursive minimax algorithm. 
        """

        # If the algorithm is at the specified depth, it calls the heuristic function on the
        # current state
        if currDepth == depth:                  
            return self.heuristic_1(state)

        # Ensures that the current player is correct just incase if any of the players skipped
        # their moves.
        current_player = state.current

        # applies a move and stores it as a new state
        new_state = self.move_minimax(move, state)

        # Flips the turn to the next player.
        if new_state.current == current_player:
            if is_Max:
                is_Max = False
            else:
                is_Max = True


        available = new_state.available_moves()

        # If game is over, calls the heuristic function on the current state
        if available == []:
            return self.heuristic_1(state)


        # If Max's turn, calls the max of value, which is supposed to represent negative infinity, and
        # the recursive call to minimax_algorithm with a depth of +1 and is_Max as False.
        if is_Max:
            value = -1000
            for move in available:
                value = max(value, self.minimax_algorithm(new_state, currDepth + 1, depth, move, False))
            return value

        # If Min's turn, calls the min of value, which is supposed to represent positive infinity, and
        # the recursive call to minimax_algorithm with a depth of +1 and is_Max as True.
        else:
            # Following line redundant
            heuristic_list = []
            value = 1000
            for move in available:
                value = min(value, self.minimax_algorithm(new_state, currDepth + 1, depth, move, True))
            return value


    def heuristic_1(self, state):
        """ 
        Basic heuristic function that returns black piecces - white pieces.
        Tries maximizing the number of current playuer pieces on the board.
        """
        return state.count("black") - state.count("white")


    def move_minimax(self, move, state):
        """
        Implements a given move on a given state and returns the new state
        """
        move = move.pair
        moveR = move[0]
        moveC = move[1]

        new_state = copy.deepcopy(state)
        move = OthelloMove(int(moveR), int(moveC), state.current)
        new_state = new_state.apply_move(move)
        return new_state



#################################################################################################################################

class AlphaBetaPruning(OthelloPlayer):
    """
    Implements the minimax algorithm but with alpha beta pruning to make the
    search more efficient. Implemeted for depth 4.
    """

    def make_move(self, state, remaining_time):
        return self.find_move(state)


    def find_move(self, state):
        """ 
        Returns a move with max/min utility
        """

        depth = 4
        available = state.available_moves()
        if state.current == "black":
            is_Max = True
        else:
            is_Max = False

        new_state = copy.deepcopy(state)

        moves_eval_list = []
        for move in available:
            moves_eval_list.append(self.AlphaBeta_algorithm(new_state, 0, depth, move, is_Max, -1000, 1000))

        if is_Max:
            return available[moves_eval_list.index(max(moves_eval_list))]
        else:
            return available[moves_eval_list.index(min(moves_eval_list))]


    def AlphaBeta_algorithm(self, state, currDepth, depth, move, is_Max, alpha, beta):
        """
        Implements the alpha beta algorithm. Returns utility of a given move.
        """

        if currDepth == depth:                      
            return self.heuristic_1(state)

        current_player = state.current
        new_state = self.move_AlphaBeta(move, state)

        if new_state.current == current_player:
            if is_Max:
                is_Max = False
            else:
                is_Max = True


        available = new_state.available_moves()

        if available == []:
            return self.heuristic_1(state)

        # If alpha is greater than or equal to beta, the branch is pruned. Alpha is the current max for the search in the
        # current branch.
        if is_Max:
            value = -1000
            heuristic_list = []
            for move in available:
                value = max(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta))
                alpha = max(value, alpha)
                if alpha >= beta:
                    return value
            return value

        # If alpha is greater than or equal to beta, the branch is pruned. Beta is the current min for the search in the
        # current branch.
        else:
            value = 1000
            heuristic_list = []
            for move in available:
                value = min(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta))
                beta = min(value, beta)
                if alpha >= beta:
                    return value
            return value


    def heuristic_1(self, state):
        return state.count("black") - state.count("white")


    def move_AlphaBeta(self, move, state):
        """
        Implements a given move on a given state and returns the new state.
        """
        move = move.pair
        moveR = move[0]
        moveC = move[1]

        new_state = copy.deepcopy(state)
        move = OthelloMove(int(moveR), int(moveC), state.current)
        new_state = new_state.apply_move(move)
        return new_state



#################################################################################################################################


class IterativeDeepeningSearchPlayer(OthelloPlayer):
    """ 
    Implements IDS for alpha beta pruning algorithm and for minimax algorithm. 
    4.9 seconds given per move if alpha beta pruning.
    """
    def make_move(self, state, remaining_time):
        return self.find_move(state)


    def find_move(self, state):
        """
        Calls the AlphaBeta_algorithm using IDS. Minimax algorithm is not called even though its
        implemented because it's slower. 
        """

        # Gets the current epoch time. 
        start_time = time.time()

        available = state.available_moves()

        if state.current == "black":
            is_Max = True
        else:
            is_Max = False

        new_state = copy.deepcopy(state)

        moves_eval_list = []

        # Keeps track of the utility of each move for the previous depth
        previous_depth_list = []

        # Calls IDS with max depth of 64. 64 is the max depth for an othello board with 0 pieces on it. 
        for depth in range(64):

            for move in available:
                moves_eval_list.append(self.AlphaBeta_algorithm(new_state, 0, depth, move, is_Max, -1000, 1000, start_time))

            # if last item in moves_eval_list is the string timeout, it breaks out of the for loop. 
            if moves_eval_list[-1] == "timeout":
                break

            # If no timeout string, it stores the moves_eval_list for the current depth to the previous_depth_list
            else:
                previous_depth_list = moves_eval_list
                moves_eval_list = []


        if is_Max:
            return available[previous_depth_list.index(max(previous_depth_list))]
        else:
            return available[previous_depth_list.index(min(previous_depth_list))]



    def minimax_algorithm(self, state, currDepth, depth, move, is_Max, start_time):
        """
        Minimax algorithm tailored for IDS. 5 seconds given per move. Is not called because slower.
        """
        
        # If the difference between current time and start time is more than 5 seconds, 
        # it returns a string timeout
        if time.time() - start_time >= 5:
            return "timeout"

        if currDepth == depth:                      
            return self.heuristic_1(state)

        current_player = state.current
        new_state = self.move_minimax(move, state)

        if new_state.current == current_player:
            if is_Max:
                is_Max = False
            else:
                is_Max = True


        available = new_state.available_moves()

        if available == []:
            return self.heuristic_1(state)


        # Uses try except to quit out of the recursive calls if a string timeout is found. Max fails to compare string to
        # an int. If no string timeout found, it returns the utility for the current move.
        if is_Max:
            value = -1000
            for move in available:
                try:
                    value = max(value, self.minimax_algorithm(new_state, currDepth + 1, depth, move, False, start_time))
                except:
                    return "timeout"
            return value

        # Uses try except to quit out of the recursive calls if a string timeout is found. Min fails to compare string to
        # an int. If no string timeout found, it returns the utility for the current move.
        else:
            heuristic_list = []
            value = 1000
            for move in available:
                try:
                    value = min(value, self.minimax_algorithm(new_state, currDepth + 1, depth, move, True, start_time))
                except:
                    return "timeout"
            return value



    def AlphaBeta_algorithm(self, state, currDepth, depth, move, is_Max, alpha, beta, start_time):
        """
        Alpha beta algorithm tailored for IDS.
        """

        if time.time() - start_time >= 4.9:
            return "timeout"

        if currDepth == depth:                      
            return self.heuristic_1(state)

        current_player = state.current

        # move_minimax is a misleading name. Should simply have been called move.
        new_state = self.move_minimax(move, state)

        if new_state.current == current_player:
            if is_Max:
                is_Max = False
            else:
                is_Max = True



        available = new_state.available_moves()

        if available == []:
            return self.heuristic_1(state)

        # Uses try except to quit out of the recursive calls if a string timeout is found. max functionfails to compare string to
        # an int. If no string timeout found, it returns the utility for the current move.
        if is_Max:
            value = -1000
            heuristic_list = []
            for move in available:
                try:
                    value = max(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time))
                    alpha = max(value, alpha)
                    if alpha >= beta:
                        return value
                except:
                    return "timeout"
            return value


        # Uses try except to quit out of the recursive calls if a string timeout is found. min function fails to compare string to
        # an int. If no string timeout found, it returns the utility for the current move.
        else:
            value = 1000
            heuristic_list = []
            for move in available:
                try:
                    value = min(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time))
                    beta = min(value, beta)
                    if alpha >= beta:
                        return value
                except:
                    return "timeout"
            return value


    def heuristic_1(self, state):
        return state.count("black") - state.count("white")


    def move_minimax(self, move, state):
        """
        Misleading name. Should have just been called move.
        """
        move = move.pair
        moveR = move[0]
        moveC = move[1]

        new_state = copy.deepcopy(state)
        move = OthelloMove(int(moveR), int(moveC), state.current)
        new_state = new_state.apply_move(move)
        return new_state



#################################################################################################################################
class Node(object):
    """
    Implements the Node class for the MontecarloTreeSearchPlayer
    """
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)


class MontecarloTreeSearchPlayer(OthelloPlayer):
    """
    Tried implementing Montecarlo search algorithm, but ran out of time.
    Apparently I deleted the stuff that I tried out, but only left this part in. 
    NOT FUNCTIONAL AT ALL. 
    """

    def make_move(self, state, remaining_time):
        return self.find_move(state)



#################################################################################################################################

class TimeRemainingIDSPlayer(OthelloPlayer):
    """
    Implements a player that used IDS and alpha beta pruning and that takes the
    time remaining into account.
    """
    def make_move(self, state, remaining_time):
        return self.find_move(state, remaining_time)



    def find_move(self, state, remaining_time):

        # Takes the time remaining into account.

        # Start game is faster. Less branching factor so less time allotted.
        if remaining_time >= 130:
            time_to_spend = 4

        # more time for midgame.
        elif remaining_time >= 70:
            time_to_spend = 5

        # End game is fast, so less time.
        elif remaining_time >= 50:
            time_to_spend = 4

        # Time running out, so player is restricted to less time so that it does not lose based on time
        # remaining.
        elif remaining_time >= 20:
            time_to_spend = 3
        else:
            time_to_spend = 2


        start_time = time.time()

        available = state.available_moves()

        if state.current == "black":
            is_Max = True
        else:
            is_Max = False

        new_state = copy.deepcopy(state)

        moves_eval_list = []
        previous_depth_list = []

        for depth in range(64):

            for move in available:
                # time_to_spend is passed in as an additional argument so that the algorithms can be ended based on time remaining. 
                moves_eval_list.append(self.AlphaBeta_algorithm(new_state, 0, depth, move, is_Max, -1000, 1000, start_time, time_to_spend))

            if moves_eval_list[-1] == "timeout":
                break
            else:
                previous_depth_list = moves_eval_list
                moves_eval_list = []


        if is_Max:
            return available[previous_depth_list.index(max(previous_depth_list))]
        else:
            return available[previous_depth_list.index(min(previous_depth_list))]



    def AlphaBeta_algorithm(self, state, currDepth, depth, move, is_Max, alpha, beta, start_time, time_to_spend):

        # If more than or equal to time_to_spend time is used, returns string timeout. 
        if time.time() - start_time >= time_to_spend:
            return "timeout"

        if currDepth == depth:                      
            return self.heuristic_1(state)

        current_player = state.current
        new_state = self.move_minimax(move, state)
        if new_state.current == current_player:
            if is_Max:
                is_Max = False
            else:
                is_Max = True


        available = new_state.available_moves()

        if available == []:
            return self.heuristic_1(state)

        if is_Max:
            value = -1000
            heuristic_list = []
            for move in available:
                try:
                    value = max(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time, time_to_spend))
                    alpha = max(value, alpha)
                    if alpha >= beta:
                        return value
                except:
                    return "timeout"
            return value

        else:
            value = 1000
            heuristic_list = [] # Maybe redundant. Check if you can
            for move in available:
                try:
                    value = min(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time, time_to_spend))
                    beta = min(value, beta)
                    if alpha >= beta:
                        return value
                except:
                    return "timeout"
            return value


    def heuristic_1(self, state):
        return state.count("black") - state.count("white")


    def move_minimax(self, move, state):
        move = move.pair
        moveR = move[0]
        moveC = move[1]

        new_state = copy.deepcopy(state)
        move = OthelloMove(int(moveR), int(moveC), state.current)
        new_state = new_state.apply_move(move)
        return new_state




#################################################################################################################################



class BetterHeuristic(OthelloPlayer):
    """
    Implements a better heuristic function with IDS and alpha beta pruning relying on remaining time.
    """

    def make_move(self, state, remaining_time):
        return self.find_move(state, remaining_time)


    def find_move(self, state, remaining_time):

        if remaining_time >= 130:
            time_to_spend = 4
        elif remaining_time >= 70:
            time_to_spend = 5
        elif remaining_time >= 50:
            time_to_spend = 4
        elif remaining_time >= 20:
            time_to_spend = 3
        else:
            time_to_spend = 2


        start_time = time.time()

        available = state.available_moves()

        if state.current == "black":
            is_Max = True
        else:
            is_Max = False

        new_state = copy.deepcopy(state)

        moves_eval_list = []
        previous_depth_list = []

        for depth in range(64):

            for move in available:
                moves_eval_list.append(self.AlphaBeta_algorithm(new_state, 0, depth, move, is_Max, -1000, 1000, start_time, time_to_spend))

            if "timeout" in moves_eval_list:
                break
            else:
                previous_depth_list = moves_eval_list
                moves_eval_list = []


        if is_Max:
            return available[previous_depth_list.index(max(previous_depth_list))]
        else:
            return available[previous_depth_list.index(min(previous_depth_list))]



    def AlphaBeta_algorithm(self, state, currDepth, depth, move, is_Max, alpha, beta, start_time, time_to_spend):

        if time.time() - start_time >= time_to_spend:
            return "timeout"

        if currDepth == depth:                      
            return self.heuristic_2(state)

        current_player = state.current
        new_state = self.move_minimax(move, state)      # correct name move_minimax
        if new_state.current == current_player:
            if is_Max:
                is_Max = False
            else:
                is_Max = True



        available = new_state.available_moves()

        if available == []:
            return self.heuristic_2(state)

        if is_Max:
            value = -1000
            heuristic_list = []
            for move in available:
                try:
                    value = max(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time, time_to_spend))
                    alpha = max(value, alpha)
                    if alpha >= beta:
                        return value
                except:
                    return "timeout"
            return value


        else:
            value = 1000
            heuristic_list = []
            for move in available:
                try:
                    value = min(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time, time_to_spend))
                    beta = min(value, beta)
                    if alpha >= beta:
                        return value
                except:
                    return "timeout"
            return value


    def heuristic_1(self, state):
        """
        Not called in this class. Redundant.
        """
        return state.count("black") - state.count("white")

    def heuristic_2(self, state):
        """
        A more informed heuristic function taking multiple strategies into consideration.
        """


        black_pieces = state.count("black")
        white_pieces = state.count("white")
        remaining_moves = 64 - black_pieces - white_pieces

        # Keeps track of the total utility through out.
        total_utility = 0


        player = state.current

        # Available moves are added to total utility. The goal is to limit the other player's moves.
        # Each move is counted as 1.
        total_utility += len(state.available_moves())

        # Each black piece on the board counts as 0.01 towards the total utility. Priority reduced for number
        # of pieces on the board.
        if player == "black":
            total_utility += black_pieces * 0.01

            # Tries getting the last move. Tries reversing the parity. Whoever gets to go in the end has an advantage
            # so takes that into account.
            if remaining_moves % 2 == 0:
                total_utility -= 5
            else:
                total_utility += 5
        else:
            # for white player, the utility is made negative.
            total_utility += white_pieces * -0.01

            # Tries getting the last move. Tries reversing the parity. Whoever gets to go in the end has an advantage
            # so takes that into account.
            if remaining_moves % 2 == 1:
                total_utility -= 5
            else:
                total_utility += 5



        # The corners in an othello board. They are important and are assigned a utility of 15.
        corners = [(0,0), (0,7), (7,0), (7,7)]
        for corner in corners:
            if state.board[corner[0]][corner[1]] == "black":
                total_utility += 15
            elif state.board[corner[0]][corner[1]] == "white":
                total_utility -= 15

        # x-squares. Squares in diagonal to the corner squares. They are bad to go to if the corner square
        # is empty. The if part for the corner squares being empty has been commented out because it did not
        # help with the winning.
        bad_moves = [(1,1), (1, 6), (6, 1), (6, 6)]

        # For remaining moves greater than 15, the player avoids going to the x_square. For the end game, it
        # does not matter.
        if remaining_moves >= 15:
            for move in bad_moves:
                if state.board[move[0]][move[1]] == "black": #and state.board[corner[0]][corner[1]] == "empty":
                    total_utility -= 5
                elif state.board[move[0]][move[1]] == "white": #and state.board[corner[0]][corner[1]] == "empty":
                    total_utility += 5
        return total_utility


    def move_minimax(self, move, state):
        move = move.pair
        moveR = move[0]
        moveC = move[1]

        new_state = copy.deepcopy(state)
        move = OthelloMove(int(moveR), int(moveC), state.current)
        new_state = new_state.apply_move(move)
        return new_state



#################################################################################################################################


class c_squares_heuristic(OthelloPlayer):
    """
    Implements c_squares as bad in the heuristic function. c squares are squares adjacent to the corner.
    """
    def make_move(self, state, remaining_time):
        return self.find_move(state, remaining_time)


    def find_move(self, state, remaining_time):

        if remaining_time >= 130:
            time_to_spend = 4
        elif remaining_time >= 70:
            time_to_spend = 5
        elif remaining_time >= 50:
            time_to_spend = 4
        elif remaining_time >= 20:
            time_to_spend = 3
        else:
            time_to_spend = 2


        start_time = time.time()

        available = state.available_moves()

        if state.current == "black":
            is_Max = True
        else:
            is_Max = False

        new_state = copy.deepcopy(state)

        moves_eval_list = []
        previous_depth_list = []

        for depth in range(64):

            for move in available:
                moves_eval_list.append(self.AlphaBeta_algorithm(new_state, 0, depth, move, is_Max, -1000, 1000, start_time, time_to_spend))

            if "timeout" in moves_eval_list:
                break
            else:
                previous_depth_list = moves_eval_list
                moves_eval_list = []


        if is_Max:
            # print("AAAAA", moves_eval_list.index(max(moves_eval_list)))
            return available[previous_depth_list.index(max(previous_depth_list))]
        else:
            # print("AAAA", moves_eval_list.index(min(moves_eval_list)))
            return available[previous_depth_list.index(min(previous_depth_list))]



    def AlphaBeta_algorithm(self, state, currDepth, depth, move, is_Max, alpha, beta, start_time, time_to_spend):

        if time.time() - start_time >= time_to_spend:
            return "timeout"

        if currDepth == depth:                      
            return self.heuristic_2(state)

        current_player = state.current
        new_state = self.move_minimax(move, state)
        if new_state.current == current_player:
            if is_Max:
                is_Max = False
            else:
                is_Max = True



        available = new_state.available_moves()

        if available == []:
            return self.heuristic_2(state)

        if is_Max:
            value = -1000
            heuristic_list = []
            for move in available:
                try:
                    value = max(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time, time_to_spend))
                    alpha = max(value, alpha)
                    if alpha >= beta:
                        return value

                except:
                    return "timeout"
            return value

        else:
            value = 1000
            heuristic_list = []
            for move in available:
                try:
                    value = min(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time, time_to_spend))
                    beta = min(value, beta)
                    if alpha >= beta:
                        return value
                except:
                    return "timeout"
            return value


    def heuristic_1(self, state):
        return state.count("black") - state.count("white")

    def heuristic_2(self, state):


        black_pieces = state.count("black")
        white_pieces = state.count("white")
        remaining_moves = 64 - black_pieces - white_pieces
        total_utility = 0
        player = state.current
        total_utility += len(state.available_moves())

        if player == "black":
            total_utility += black_pieces * 0.01
        else:
            total_utility += white_pieces * -0.01


        corners = [(0,0), (0,7), (7,0), (7,7)]

        for corner in corners:
            if state.board[corner[0]][corner[1]] == "black":
                total_utility += 15
            elif state.board[corner[0]][corner[1]] == "white":
                total_utility -= 15

        # x-squares and c_squares. 
        bad_moves_1 = [(1,1), (1,0), (0,1)]
        bad_moves_2 = [(1,6), (0,6), (1,7)]
        bad_moves_3 = [(6,1), (6,0), (7,1)]
        bad_moves_4 = [(6,6), (6,7), (7,6)]

        # For remaining moves greater than 10, assigns a negative utility to these moves. For end game,
        # does not matter because it can calculate the entire game tree. 
        if remaining_moves >= 10:
            for i in range(4):
                corner = corners[i]
                if state.board[corner[0]][corner[1]] == "empty":
                    for move in bad_moves_1:
                        if state.board[move[0]][move[1]] == "black":
                            total_utility -= 5
                        elif state.board[move[0]][move[1]] == "white":
                            total_utility += 5
                    for move in bad_moves_2:
                        if state.board[move[0]][move[1]] == "black":
                            total_utility -= 5
                        elif state.board[move[0]][move[1]] == "white":
                            total_utility += 5
                    for move in bad_moves_3:
                        if state.board[move[0]][move[1]] == "black":
                            total_utility -= 5
                        elif state.board[move[0]][move[1]] == "white":
                            total_utility += 5
                    for move in bad_moves_4:
                        if state.board[move[0]][move[1]] == "black":
                            total_utility -= 5
                        elif state.board[move[0]][move[1]] == "white":
                            total_utility += 5
        return total_utility



    def move_minimax(self, move, state):
        move = move.pair
        moveR = move[0]
        moveC = move[1]

        new_state = copy.deepcopy(state)
        move = OthelloMove(int(moveR), int(moveC), state.current)
        new_state = new_state.apply_move(move)
        return new_state


#################################################################################################################################



class Superior(OthelloPlayer):
    """ 
    Was supposed to be the most superior player, but it's plagued with bugs. Needs debugging.
    Implements a few more ideas.
    """

    def make_move(self, state, remaining_time):
        return self.find_move(state, remaining_time)


    def find_move(self, state, remaining_time):

        available = state.available_moves()
        black_pieces = state.count("black")
        white_pieces = state.count("white")

        remaining_moves = 64 - black_pieces - white_pieces

        # First piece does not matter, so choses the first available move. All first moves lead to
        # symmetry in different directions.
        # CITE: http://radagast.se/othello/Help/strategy.html
        if black_pieces + white_pieces == 4:
            return available[0]

        # End game. More time provided so it can get the complete Othello game tree for the remaining moves.
        if remaining_moves < 10:
            time_to_spend = 7
            if remaining_time < 15:
                time_to_spend = 5
            if remaining_time < 5:
                time_to_spend = 1

        # For the first 3 moves, less time provided because tree is less branches and quicker to go through. Also
        # these moves do not have alot of affect on the end game.
        elif remaining_moves > 53:
            time_to_spend = 3

        # Time spent during midgame.
        else:
            time_to_spend = 4.2


        start_time = time.time()



        if state.current == "black":
            is_Max = True
        else:
            is_Max = False

        new_state = copy.deepcopy(state)

        moves_eval_list = []
        previous_depth_list = []

        # Slight optimization to start IDS at 4. The idea was that I would prevent my IDS from calculating again for
        # depth 0,1,2,3 since it always made it to depth 4. This idea is not implemented for the the end game because 
        # low branching factor.
        if remaining_moves > 20:
            start = 3
        else:
            start = 0



        for depth in range(start, remaining_moves + 1):

            for move in available:
                moves_eval_list.append(self.AlphaBeta_algorithm(new_state, 0, depth, move, is_Max, -1000, 1000, start_time, time_to_spend))

            if "timeout" in moves_eval_list:
                break
            else:
                previous_depth_list = moves_eval_list
                moves_eval_list = []


        if is_Max:
            return available[previous_depth_list.index(max(previous_depth_list))]
        else:
            return available[previous_depth_list.index(min(previous_depth_list))]



    def AlphaBeta_algorithm(self, state, currDepth, depth, move, is_Max, alpha, beta, start_time, time_to_spend):

        black_pieces = state.count("black")
        white_pieces = state.count("white")

        remaining_moves = 64 - black_pieces - white_pieces

        end_game = False

        # classifies end game as moves < 10.
        if remaining_moves < 10:
            end_game = True


        if time.time() - start_time >= time_to_spend:
            return "timeout"

        # Changes the heuristic for the end game. For end game since it can calculate the remaining game tree, the
        # heuristic just tries to get the maximum number of pieces.
        if currDepth == depth:                      
            if end_game == True:
                return self.heuristic_1
            return self.heuristic_2(state)

        current_player = state.current
        new_state = self.move_minimax(move, state)
        if new_state.current == current_player:
            if is_Max:
                is_Max = False
            else:
                is_Max = True



        available = new_state.available_moves()

        # When game is over: Changes the heuristic for the end game. For end game since it can calculate the remaining game tree, the
        # heuristic just tries to get the maximum number of pieces.
        if available == []:
            if end_game == True:
                return self.heuristic_1
            return self.heuristic_2(state)

        if is_Max:
            value = -1000
            heuristic_list = []
            for move in available:
                try:
                    value = max(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time, time_to_spend))
                    alpha = max(value, alpha)
                    if alpha >= beta:
                        return value

                except:
                    return "timeout"
            return value

        else:
            value = 1000
            heuristic_list = []
            for move in available:
                try:
                    value = min(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time, time_to_spend))
                    beta = min(value, beta)
                    if alpha >= beta:
                        return value
                except:
                    return "timeout"
            return value


    def heuristic_1(self, state):
        return state.count("black") - state.count("white")

    def heuristic_2(self, state):


        black_pieces = state.count("black")
        white_pieces = state.count("white")
        remaining_moves = 64 - black_pieces - white_pieces
        total_utility = 0
        player = state.current
        total_utility += len(state.available_moves())

        if player == "black":
            total_utility += black_pieces * 0.01

            # Tries getting the last move 
            if remaining_moves % 2 == 0:
                total_utility -= 5
            else:
                total_utility += 5
        else:
            total_utility += white_pieces * -0.01

            # Tries getting the last move
            if remaining_moves % 2 == 1:
                total_utility -= 5
            else:
                total_utility += 5




        corners = [(0,0), (0,7), (7,0), (7,7)]

        for corner in corners:
            if state.board[corner[0]][corner[1]] == "black":
                total_utility += 15
            elif state.board[corner[0]][corner[1]] == "white":
                total_utility -= 15

        # x-squares
        bad_moves = [(1,1), (1, 6), (6, 1), (6, 6)]

        if remaining_moves >= 15:
            for move in bad_moves:
                if state.board[move[0]][move[1]] == "black": #and state.board[corner[0]][corner[1]] == "empty":
                    total_utility -= 5
                elif state.board[move[0]][move[1]] == "white": #and state.board[corner[0]][corner[1]] == "empty":
                    total_utility += 5

        return total_utility


    def move_minimax(self, move, state):
        move = move.pair
        moveR = move[0]
        moveC = move[1]

        new_state = copy.deepcopy(state)
        move = OthelloMove(int(moveR), int(moveC), state.current)
        new_state = new_state.apply_move(move)
        return new_state



#################################################################################################################################


class Move_ordering(OthelloPlayer):
    """
    Tries implementing move ordering for IDS to make it faster. Buggy.
    """
    def make_move(self, state, remaining_time):
        return self.find_move(state, remaining_time)


    def find_move(self, state, remaining_time):

        if remaining_time >= 130:
            time_to_spend = 4
        elif remaining_time >= 70:
            time_to_spend = 5
        elif remaining_time >= 50:
            time_to_spend = 4
        elif remaining_time >= 20:
            time_to_spend = 3
        else:
            time_to_spend = 2


        start_time = time.time()

        available = state.available_moves()

        if state.current == "black":
            is_Max = True
        else:
            is_Max = False

        new_state = copy.deepcopy(state)

        moves_eval_list = []
        previous_depth_list = []
        previous_depth_list_2 = []
        available_backup = available

        for depth in range(64):

            for move in available:
                moves_eval_list.append(self.AlphaBeta_algorithm(new_state, 0, depth, move, is_Max, -1000, 1000, start_time, time_to_spend))

            if "timeout" in moves_eval_list:
                print("AAAAA")
                for _ in range(10):
                    print(depth)
                print(available)
                print(previous_depth_list)
                break
            else:
                previous_depth_list_2 = previous_depth_list
                previous_depth_list = moves_eval_list
                moves_eval_list = []

            print("AVAILABLE: ", available)
            previous_depth_list_2 = previous_depth_list
            available = self.order_moves(available, previous_depth_list)


        if is_Max:
            # print("AAAAA", moves_eval_list.index(max(moves_eval_list)))
            return available_backup[previous_depth_list.index(max(previous_depth_list_2))]
        else:
            # print("AAAA", moves_eval_list.index(min(moves_eval_list)))
            return available_backup[previous_depth_list.index(min(previous_depth_list_2))]


    def order_moves(self, available, previous_depth_list):
        """
        Orders moves based on their utility. Somewhere along one of the lists gets messed up and ends up
        being empty.
        """

        available_ordered = []
        # print("aaa")
        # print("prev: ", previous_depth_list)
        # print("avail: ", available)
        # print("bbb")
        iterations = len(available)


        available_2 = available
        print("AVAILABLE ORDER MOVES BEGINNING: ", available_2, "previous_depth_list: ", previous_depth_list)
        for _ in range(iterations):
            print("prev: ", previous_depth_list)
            print("avail: ", available)
            if len(previous_depth_list) > 1:
                pop_index = previous_depth_list.index(max(previous_depth_list))
            else:
                pop_index = 0
            print("Pop index: ", pop_index)
            available_ordered.append(available.pop(pop_index))
            previous_depth_list.pop(pop_index)
            print("AVAILABLE ORDERED: ", available_ordered)

        print("AVAILABE ORDERED END: ", available_ordered)
        return available_ordered




    def AlphaBeta_algorithm(self, state, currDepth, depth, move, is_Max, alpha, beta, start_time, time_to_spend):

        if time.time() - start_time >= time_to_spend:
            return "timeout"

        if currDepth == depth:                      
            return self.heuristic_2(state)

        current_player = state.current
        new_state = self.move_minimax(move, state) 
        if new_state.current == current_player:
            if is_Max:
                is_Max = False
            else:
                is_Max = True



        available = new_state.available_moves()

        if available == []:
            return self.heuristic_2(state)

        if is_Max:
            value = -1000
            heuristic_list = []
            for move in available:
                try:
                    value = max(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time, time_to_spend))
                    alpha = max(value, alpha)
                    if alpha >= beta:
                        return value

                except:
                    return "timeout"
            return value

        else:
            value = 1000
            heuristic_list = []
            for move in available:
                try:
                    value = min(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time, time_to_spend))
                    beta = min(value, beta)
                    if alpha >= beta:
                        return value
                except:
                    return "timeout"
            return value


    def heuristic_1(self, state):
        return state.count("black") - state.count("white")


    def heuristic_2(self, state):
        black_pieces = state.count("black")
        white_pieces = state.count("white")
        remaining_moves = 64 - black_pieces - white_pieces
        total_utility = 0
        player = state.current
        total_utility += len(state.available_moves())

        if player == "black":
            total_utility += black_pieces * 0.01

            # Tries getting the last move 
            if remaining_moves % 2 == 0:
                total_utility -= 5
            else:
                total_utility += 5
        else:
            total_utility += white_pieces * -0.01

            # Tries getting the last move
            if remaining_moves % 2 == 1:
                total_utility -= 5
            else:
                total_utility += 5


        corners = [(0,0), (0,7), (7,0), (7,7)]

        for corner in corners:
            if state.board[corner[0]][corner[1]] == "black":
                total_utility += 15
            elif state.board[corner[0]][corner[1]] == "white":
                total_utility -= 15

        # x-squares
        bad_moves = [(1,1), (1, 6), (6, 1), (6, 6)]

        if remaining_moves >= 15:
            for move in bad_moves:
                if state.board[move[0]][move[1]] == "black": #and state.board[corner[0]][corner[1]] == "empty":
                    total_utility -= 5
                elif state.board[move[0]][move[1]] == "white": #and state.board[corner[0]][corner[1]] == "empty":
                    total_utility += 5

        return total_utility


    def move_minimax(self, move, state):
        move = move.pair
        moveR = move[0]
        moveC = move[1]

        new_state = copy.deepcopy(state)
        move = OthelloMove(int(moveR), int(moveC), state.current)
        new_state = new_state.apply_move(move)
        return new_state



#################################################################################################################################


class ExperimentalPlayer1(OthelloPlayer):
    """
    Experimental player created to implement minimzer player at the beginning of the game so that 
    the options for the other players were reduced. Idea abandoned after I started relying on
    remaining moves.
    """

    def make_move(self, state, remaining_time):
        return self.experimental_play(state)

    def experimental_play(self, state):
        available = state.available_moves()

        score_change = []
        if state.count("black") + state.count("white") <= 6:
            for move in available:
                move = move.pair
                moveR = move[0]
                moveC = move[1]

                new_state = copy.deepcopy(state)
                move = OthelloMove(int(moveR), int(moveC), state.current)
                new_state = new_state.apply_move(move)
                score_change.append(new_state.count(state.current))


            # The move with the min score for the current player is used
            index_min = score_change.index(min(score_change))
            return available[index_min]


#################################################################################################################################

class TournamentPlayer(OthelloPlayer):
    """
    Player to perform against other players in class. Basically the same as BetterHeuristic but with
    Parity removed from the heuristic function beccause it was hurting my wins against a random player.
    """
    
    def make_move(self, state, remaining_time):
        return self.find_move(state, remaining_time)


    def find_move(self, state, remaining_time):

        if remaining_time >= 130:
            time_to_spend = 4
        elif remaining_time >= 70:
            time_to_spend = 5
        elif remaining_time >= 50:
            time_to_spend = 4
        elif remaining_time >= 20:
            time_to_spend = 3
        else:
            time_to_spend = 2


        start_time = time.time()

        available = state.available_moves()
        if state.current == "black":
            is_Max = True
        else:
            is_Max = False

        new_state = copy.deepcopy(state)

        moves_eval_list = []
        previous_depth_list = []

        for depth in range(64):

            for move in available:
                moves_eval_list.append(self.AlphaBeta_algorithm(new_state, 0, depth, move, is_Max, -1000, 1000, start_time, time_to_spend))

            if "timeout" in moves_eval_list:
                break
            else:
                previous_depth_list = moves_eval_list
                moves_eval_list = []


        if is_Max:
            return available[previous_depth_list.index(max(previous_depth_list))]
        else:
            return available[previous_depth_list.index(min(previous_depth_list))]



    def AlphaBeta_algorithm(self, state, currDepth, depth, move, is_Max, alpha, beta, start_time, time_to_spend):

        if time.time() - start_time >= time_to_spend:
            return "timeout"

        if currDepth == depth:                      
            return self.heuristic_2(state)

        current_player = state.current
        new_state = self.move_minimax(move, state)
        if new_state.current == current_player:
            if is_Max:
                is_Max = False
            else:
                is_Max = True


        available = new_state.available_moves()

        if available == []:
            return self.heuristic_2(state)

        if is_Max:
            value = -1000
            heuristic_list = []
            for move in available:
                try:
                    value = max(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time, time_to_spend))
                    alpha = max(value, alpha)
                    if alpha >= beta:
                        return value
                except:
                    return "timeout"
            return value

        else:
            value = 1000
            heuristic_list = []
            for move in available:
                try:
                    value = min(value, self.AlphaBeta_algorithm(new_state, currDepth + 1, depth, move, False, alpha, beta, start_time, time_to_spend))
                    beta = min(value, beta)
                    if alpha >= beta:
                        return value
                except:
                    return "timeout"
            return value


    def heuristic_1(self, state):
        return state.count("black") - state.count("white")


    def heuristic_2(self, state):
        black_pieces = state.count("black")
        white_pieces = state.count("white")
        remaining_moves = 64 - black_pieces - white_pieces
        total_utility = 0
        player = state.current
        total_utility += len(state.available_moves())

        if player == "black":
            total_utility += black_pieces * 0.01
        else:
            total_utility += white_pieces * -0.01


        corners = [(0,0), (0,7), (7,0), (7,7)]

        for corner in corners:
            if state.board[corner[0]][corner[1]] == "black":
                total_utility += 15
            elif state.board[corner[0]][corner[1]] == "white":
                total_utility -= 15

        # x-squares
        bad_moves = [(1,1), (1, 6), (6, 1), (6, 6)]

        if remaining_moves >= 15:
            for move in bad_moves:
                if state.board[move[0]][move[1]] == "black": #and state.board[corner[0]][corner[1]] == "empty":
                    total_utility -= 5
                elif state.board[move[0]][move[1]] == "white": #and state.board[corner[0]][corner[1]] == "empty":
                    total_utility += 5

        return total_utility


    def move_minimax(self, move, state):
        move = move.pair
        moveR = move[0]
        moveC = move[1]

        new_state = copy.deepcopy(state)
        move = OthelloMove(int(moveR), int(moveC), state.current)
        new_state = new_state.apply_move(move)
        return new_state

################################################################################

def main():
    """Plays the game."""

    winner_list = []
    time = []
    for _ in range(50):
        black_player = TournamentPlayer("black")
        white_player = RandomPlayer("white")

        game = OthelloGame(black_player, white_player, verbose=True)

        winner = game.play_game_timed()

        time_used = 150 - game.black_time
        time.append(time_used)


        winner_list.append(winner)

        for _ in range(10):
            print(len(winner_list))
            print("White: ", winner_list.count("white"))
            print("Black: ", winner_list.count("black"))
            print("Draw: ", winner_list.count("draw"))


    print(time)
    print("Mean time: ", mean(time))
    print("Max time: ", max(time))
    print("Min time: ", min(time))

    print(len(winner_list))
    print("White: ", winner_list.count("white"))
    print("Black: ", winner_list.count("black"))
    print("Draw: ", winner_list.count("draw"))


    ###### Use this method if you want to use a HumanPlayer
    # winner = game.play_game()


    if not game.verbose:
        print("Winner is", winner)


if __name__ == "__main__":
    main()
