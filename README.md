# ChessAI

In this project I tried to create a Deep Learning based AI that learns to play the game of chess by regressing the stockfish evaluation score of chess positions and using this score to make informed decisions in a MinMax search for the game tree. For this I trained a DNN with residual connections on 1 million chess games that were played online on lichess and annotated by the stockfish engine.

The engine places the different positions on a zero centered scale that indicates the tendency of the positional advantage a player has. 
This scale related to the so called "centi-pawns" scale. With this, we not only consider a bare advantage in terms of piece count but also advantage in terms position.


I used [python-chess](https://python-chess.readthedocs.io/en/latest/) as a chess framework in order to fully concentrate this project on the decision making that goes into selecting the right rule-conforming move.
If you want to build your own chess-related project in python I recommend to check it out as it is easy to use and well documented.


# Approach

To play the game of chess one only needs to be able to understand what the rules are and what a good position is.
From this we can define a engine that optimizes the position by playing move that give your player an advantage while following the rules.

As indicated above we don't need to care about the rules to much as python-chess takes care of move generation and other concepts like mate, stale-mate, etc.

Therefore we only need to be able to identify what a good position is and how we can leverage the current position to translate the state of the board to a state in which we are more likely to win.

For this we need to follow 5 steps:

- get data
- create dataset
- define model
- train model
- evaluate results and play some games of chess

# Data

The most commonly used data format to represent chess games is the [Portable Game Notation (PGN)](https://en.wikipedia.org/wiki/Portable_Game_Notation). Idealy we collect .pgn data with engine annotated position evaluations as this requires substantial comutational effort to do on our own.

Luckily there are many websites providing such datasets. I want to reference the [data provided by LiChess](https://database.lichess.org/) as this is what I used to train in this project. They host over 1 TB or 3.5 billion chess games for free to use. 
Not every position is annotated in their database but still about 10% is, which is plenty enough.

If you are not familiar with the [PGN format](https://en.wikipedia.org/wiki/Portable_Game_Notation) take a look now.
You will notice that a Deep Learning model won't be able to use this data without any preprocessing (or maybe some NLP magic that would be overengineered).

Therefore I had to extract the individual games/positions from the PGN and transform it into a usable format.

See below for a visualisation of how this is done (one hot encoding the pieces on a 8 by 8 grid for each different piece type).

| source | <img src="./imgs/figs/white/pawn.svg"> | <img src="./imgs/figs/white/rook.svg"> | <img src="./imgs/figs/white/knight.svg"> | <img src="./imgs/figs/white/bishop.svg"> | <img src="./imgs/figs/white/king.svg"> | <img src="./imgs/figs/white/queen.svg"> |  <img src="./imgs/figs/black/pawn.svg"> | <img src="./imgs/figs/black/rook.svg"> | <img src="./imgs/figs/black/knight.svg"> | <img src="./imgs/figs/black/bishop.svg"> | <img src="./imgs/figs/black/king.svg"> | <img src="./imgs/figs/black/queen.svg"> |
|--|--|--|--|--|--|--|--|--|--|--|--|--|
| <img src="./imgs/figs/board.svg"> | <img src="./imgs/figs/white/pawn_map.svg"> | <img src="./imgs/figs/white/rook_map.svg"> | <img src="./imgs/figs/white/knight_map.svg"> | <img src="./imgs/figs/white/bishop_map.svg"> | <img src="./imgs/figs/white/king_map.svg"> | <img src="./imgs/figs/white/queen_map.svg"> | <img src="./imgs/figs/black/pawn_map.svg"> | <img src="./imgs/figs/black/rook_map.svg"> | <img src="./imgs/figs/black/knight_map.svg"> | <img src="./imgs/figs/black/bishop_map.svg"> | <img src="./imgs/figs/black/king_map.svg"> | <img src="./imgs/figs/black/queen_map.svg"> |


# Dataset

# Model

Below one can see the network architecture I used.

# Training

# Results

The following graph displays the training process as one can see the score has a average L1 test loss of xyz.


One can see that the engine learns and improves its play by looking at different positions.


Allthough it could improve by a lot if the data used to train was more randomly selected.
It operates on complete games that are not played at a very high ELO ranking and therefore the positions are not i.i.d

SOTA results like Deepminds AlphaZero obviously rely on substantial ressources which one can't easily access.

But still it was possible to create a simple DNN that learns how to play the game of chess on a low level.


## Observations

The engine learned basic principles in chess that beginners a thaught like:

- rooks should be connected
- knights shouldn't be on the borders of the board
- center control is important
- keep pieces protected
- pushing pawns makes sense
- keep the king save
- don't make bad trades

For sure it doesn't always play according to these rules and still makes major mistakes but still: by observing some games it played against itself, a notion of these principles can be interpreted into the engine's decision-making / moves it makes