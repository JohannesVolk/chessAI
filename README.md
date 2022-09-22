# ChessAI

In this project I tried to create a Deep Learning based AI that learns to play the game of chess by regressing the stockfish evaluation score of chess positions. For this I trained a DNN with residual connections on 1 million chess games that were played online on lichess and annotated by the stockfish engine.

The engine places the different positions on a zero centered scale that indicates the tendency of the positional advantage a player has. 
This scale is quantified in so called "centi-pawns" which are the equivalent of a hundredth pawn. In this way not only a bare advantage in terms of piece count but also advantage in terms position is considered.

Below one can see the network architecture I used.


The following graph displays the training process as one can see the score has a average L1 test loss of xyz. 