# Multipack

4k context, bsz =4,
each character represents 256 tokens
X represents a padding token

```
   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
[[ A A A A A A A A A A A ]
   B B B B B B ]
   C C C C C C C ]
   D D D D ]]

[[ E E E E E E E E ]
 [ F F F F ]
 [ G G G ]
 [ H H H H ]]

[[ I I I ]
 [ J J J ]
 [ K K K K K]
 [ L L L ]]
```

after padding to longest input in each step
```
   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
[[ A A A A A A A A A A A ]
   B B B B B B X X X X X X ]
   C C C C C C C X X X X ]
   D D D D X X X X X X X ]]

[[ E E E E E E E E ]
 [ F F F F X X X X ]
 [ G G G X X X X X ]
 [ H H H H X X X X ]]

[[ I I I X X ]
 [ J J J X X ]
 [ K K K K K ]
 [ L L L X X ]]
```

w packing ( note it's the same effective number of tokens per step, but a true bsz of 1)
```
   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
[[ A A A A A A A A A A A B B B B B
   B C C C C C C C D D D D E E E E
   E E E E F F F F F G G G H H H H
   I I I J J J J K K K K K L L L X ]]
```
