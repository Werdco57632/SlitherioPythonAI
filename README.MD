The entry point is slitherio\_render.py, run that to actually start the game.



you can set the arguments at around line 150 or use command line arguments:

&#x09;-p / --playerless (make all snakes AI / no player)

&#x09;-g / --headless (makes it run heedlessly and also no player; the g is for... graphicsless? idk, -h was reserved)



put any new algorithms as a new class in AI\_algorithms, just make a copy of Circle\_bot



in the Slitherio class in game\_logic, there are a bunch of game parameters, these can be adjusted if the game feels off, the AI can access these with self.game.PARAMETER\_HERE. It can also access runtime variables from objects like: self.state.snake\_list\[player\_id].length

