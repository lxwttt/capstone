# Structure planing 

## Input
- Armors:
    - 6 axis of armor

## Process
- estimate 6axis dynamics of armor

## Output (active output publisher)
- x_center, y_center, z_center
- r_1, r_2
- z_1, z_2
- vx


// dynamic model


// [ x, y, z, theta, vx, vy, vz, omega, r ]  [ x, y, z, theta]
// [ 0, 1, 2,   3,   4,  5,  6,    7,   8 ]  [ 0, 1, 2,   3  ]

// center模型


// [ x, y, vx, vy ]    [ x, y ]


// [ 0, 1, 2,  3  ]    [ 0, 1 ]

// Omega模型


// [ theta, omega, beta ]    [ theta ]
// [   0,     1,    2   ]    [   0   ]