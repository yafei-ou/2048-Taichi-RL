# 2048-Taichi-RL

Using Taichi Lang for implementing massive parallel RL training envs for the game "2048".

This is something I randomly came up with just for experimental purposes.

90% written by ChatGPT 4o (2024-12-25). No quality guarantees.

## Evaluation

After some not well-thought training trials, I have got the following results. The agent reaches 2048 in over 60% of its gameplays.

| Max Tile  | Occurrences |
| --------- | ----------- |
| Tile 4096 | 51          |
| Tile 2048 | 6388        |
| Tile 1024 | 3221        |
| Tile 512  | 315         |
| Tile 256  | 23          |
| Tile 128  | 2           |
| Tile 64   | 1           |

Completed episodes: 10001