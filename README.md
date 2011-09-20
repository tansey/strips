# Stanford Research Institute Problem Solver (STRIPS)
STRIPS is a First-Order Logic (FOL) language with an associated linear (total order) solver. Note that linear solvers have a few well-known issues, such as not dealing well with [http://en.wikipedia.org/wiki/Sussman_Anomaly](Sussman's Anomaly). Nevertheless, when learning planning algorithms, most students learn STRIPS first.

## Problem Description
The code implements the STRIPS problem description language. The first line is the initial state of the world:

    Initial state: At(A), Level(low), BoxAt(C), BananasAt(B)

Each initial state argument must be a literal. Currently, starting literals with upper case is allowed but may be changed in the future. It's advised that you follow the convention of literals being all lowercase and variables beginning with an upper case letter.

The second line describes the goal state you want the world to be in:

    Goal state:    Have(Bananas)

Again, only literals are allowed at the moment. This may change in the future.

The third line should declare the start of your action methods:

    Actions:

Each action has three lines: a declaration, then a set of preconditions, and finally a set of postconditions. The declaration line specifies the name of the method and its parameters; all action parameters are variables:

    // move from X to Y
    Move(X, Y)
    Preconditions:  At(X), Level(low)
    Postconditions: !At(X), At(Y)

The preconditions line contains states that the world must be in before the action can be called. In contrast, the postconditions line describes the effect of calling the action. You are able to mix variables and literals in the argument types for both preconditions and postconditions.

Note that the app follows a closed-world assumption; that is, any facts about the world not declared in the initial state are considered to be false. The parser is clever enough that you don't need to declare all your literals in the initial state, as long as you use them somewhere in the problem description. You can declare negations by using an exclamation mark (!) immediately before the name of a condition.

## Linear Solver
The strips solver is a simple linear solver. In planning literature, this is sometimes called a total order solver since it assumes all steps come one after the other with a strict ordering. It works by looking at the goal state and searching for a grounded action (one with all variables replaced with literals) that will move the world one step closer to the desired goal state. When it finds an action, it adds it to the plan and considers the current action to be the goal state. The algorithm keeps working backwards via depth-first search until it either arrives at the goal state or exhausts all possible plans and gives up.

## Running the example
To run the example problem, simply execute the following:

    python strips.py test_strips.txt

The app should read in the test file and find a plan that enables the monkey to grab the bananas.




