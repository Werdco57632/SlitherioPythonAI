The entry point is slitherio\_render.py, run that to actually start the game.


(that file will have CMD args for whether to run headless or allow player in the game later,

for now its in the file at around line 150)



put any new algorithms as a new class in AI\_algorithms, just make a copy of Circle\_bot



in the Slitherio class in game\_logic, there are a bunch of game parameters, these can be adjusted if the game feels off, the AI can access these with self.game.PARAMETER\_HERE. It can also access runtime variables from objects like: self.state.snake\_list\[player\_id].length







