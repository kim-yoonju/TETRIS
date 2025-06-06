print(f"테트리스")

map = [[0 for _ in range(10)] for _ in range(20)]

for row in map:
    print(*row, sep=' ')

blocks = {
    "I": [[0, 1, 0, 0,
           0, 1, 0, 0,
           0, 1, 0, 0,
           0, 1, 0, 0],
          [0, 0, 0, 0,
           1, 1, 1, 1,
           0, 0, 0, 0,
           0, 0, 0, 0],
          [0, 1, 0, 0,
           0, 1, 0, 0,
           0, 1, 0, 0,
           0, 1, 0, 0],
          [0, 0, 0, 0,
           1, 1, 1, 1,
           0, 0, 0, 0,
           0, 0, 0, 0],
    ],
    "O": [[1, 1,
           1, 1],
          [1, 1,
           1, 1],
          [1, 1,
           1, 1],
          [1, 1,
           1, 1],
    ],
    "Z": [[1, 1, 0,
           0, 1, 1,
           0, 0, 0],
          [0, 0, 1,
           0, 1, 1,
           0, 1, 0],
          [1, 1, 0,
           0, 1, 1,
           0, 0, 0],
          [0, 0, 1,
           0, 1, 1,
           0, 1, 0],
    ],
    "S": [[0, 1, 1,
           1, 1, 0,
           0, 0, 0],
          [0, 1, 0,
           0, 1, 1,
           0, 0, 1],
          [0, 1, 1,
           1, 1, 0,
           0, 0, 0],
          [0, 1, 0,
           0, 1, 1,
           0, 0, 1],
    ],
    "J": [[0, 1, 0,
           0, 1, 0,
           1, 1, 0],
          [1, 0, 0,
           1, 1, 1,
           0, 0, 0],
          [0, 1, 1,
           0, 1, 0,
           0, 1, 0],
          [0, 0, 0,
           1, 1, 1,
           0, 0, 1],
    ],
    "L": [[0, 1, 0,
           0, 1, 0,
           0, 1, 1],
          [1, 1, 1,
           1, 0, 0,
           0, 0, 0],
          [1, 1, 0,
           0, 1, 0,
           0, 1, 0],
          [0, 0, 1,
           1, 1, 1,
           0, 0, 0],
    ],
    "T": [[1, 1, 1,
           0, 1, 0,
           0, 0, 0],
          [0, 1, 0,
           1, 1, 0,
           0, 1, 0],
          [0, 1, 0,
           1, 1, 1,
           0, 0, 0],
          [0, 1, 0,
           0, 1, 1,
           0, 1, 0],
    ],
}